/* GStreamer
 * Copyright (C) 1999,2000 Erik Walthinsen <omega@cse.ogi.edu>
 *               2000 Wim Taymans <wtay@chello.be>
 *               2004 Thomas Vander Stichele <thomas@apestaart.org>
 *
 * gst-launch.c: tool to launch GStreamer pipelines from the command line
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <locale.h>             /* for LC_ALL */
#include <gst/gst.h>


static gboolean is_live = FALSE;

gboolean run_pipeline (GstElement *pipeline);


/* returns TRUE if there was an error or we caught a keyboard interrupt. */
static gboolean
event_loop (GstElement * pipeline, gboolean blocking, GstState target_state)
{
  GstBus *bus;
  GstMessage *message = NULL;
  gboolean res = FALSE;
  gboolean buffering = FALSE;

  bus = gst_element_get_bus (GST_ELEMENT (pipeline));

  while (TRUE) {
    message = gst_bus_poll (bus, GST_MESSAGE_ANY, blocking ? -1 : 0);

    /* if the poll timed out, only when !blocking */
    if (message == NULL)
      goto exit;

    switch (GST_MESSAGE_TYPE (message)) {
      case GST_MESSAGE_NEW_CLOCK:
      {
        GstClock *clock;

        gst_message_parse_new_clock (message, &clock);

        GST_INFO ("New clock: %s", (clock ? GST_OBJECT_NAME (clock) : "NULL"));
        break;
      }
      case GST_MESSAGE_EOS:
        GST_INFO ("Got EOS from element \"%s\".",
            GST_STR_NULL (GST_ELEMENT_NAME (GST_MESSAGE_SRC (message))));
        goto exit;
      case GST_MESSAGE_TAG:
        break;
      case GST_MESSAGE_INFO:{
        GError *gerror;
        gchar *debug;
        gchar *name = gst_object_get_path_string (GST_MESSAGE_SRC (message));

        gst_message_parse_info (message, &gerror, &debug);
        if (debug) {
          g_print ("INFO:\n%s\n", debug);
        }
        g_error_free (gerror);
        g_free (debug);
        g_free (name);
        break;
      }
      case GST_MESSAGE_WARNING:{
        GError *gerror;
        gchar *debug;
        gchar *name = gst_object_get_path_string (GST_MESSAGE_SRC (message));

        /* dump graph on warning */
        GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (pipeline),
            GST_DEBUG_GRAPH_SHOW_ALL, "gst-launch.warning");

        gst_message_parse_warning (message, &gerror, &debug);
        GST_INFO ("WARNING: from element %s: %s", name, gerror->message);
        if (debug) {
          GST_INFO ("Additional debug info: %s", debug);
        }
        g_error_free (gerror);
        g_free (debug);
        g_free (name);
        break;
      }
      case GST_MESSAGE_ERROR:{
        GError *gerror;
        gchar *debug;

        /* dump graph on error */
        GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (pipeline),
            GST_DEBUG_GRAPH_SHOW_ALL, "gst-launch.error");

        gst_message_parse_error (message, &gerror, &debug);
        gst_object_default_error (GST_MESSAGE_SRC (message), gerror, debug);
        g_error_free (gerror);
        g_free (debug);
        /* we have an error */
        res = TRUE;
        goto exit;
      }
      case GST_MESSAGE_STATE_CHANGED:{
        GstState old, new, pending;

        gst_message_parse_state_changed (message, &old, &new, &pending);

        /* we only care about pipeline state change messages */
        if (GST_MESSAGE_SRC (message) != GST_OBJECT_CAST (pipeline))
          break;

        /* ignore when we are buffering since then we mess with the states
         * ourselves. */
        if (buffering) {
          GST_INFO("Prerolled, waiting for buffering to finish...");
          break;
        }

        /* if we reached the final target state, exit */
        if (target_state == GST_STATE_PAUSED && new == target_state)
          goto exit;

        /* else not an interesting message */
        break;
      }
      case GST_MESSAGE_BUFFERING:{
        gint percent;

        gst_message_parse_buffering (message, &percent);
        GST_INFO ("buffering... %d", percent);

        /* no state management needed for live pipelines */
        if (is_live)
          break;

        if (percent == 100) {
          /* a 100% message means buffering is done */
          buffering = FALSE;
          /* if the desired state is playing, go back */
          if (target_state == GST_STATE_PLAYING) {
            GST_INFO("Done buffering, setting pipeline to PLAYING ...");
            gst_element_set_state (pipeline, GST_STATE_PLAYING);
          } else
            goto exit;
        } else {
          /* buffering busy */
          if (buffering == FALSE && target_state == GST_STATE_PLAYING) {
            /* we were not buffering but PLAYING, PAUSE  the pipeline. */
            GST_INFO("Buffering, setting pipeline to PAUSED ...");
            gst_element_set_state (pipeline, GST_STATE_PAUSED);
          }
          buffering = TRUE;
        }
        break;
      }
      case GST_MESSAGE_APPLICATION:{
        const GstStructure *s;

        s = gst_message_get_structure (message);

        if (gst_structure_has_name (s, "GstLaunchInterrupt")) {
          /* this application message is posted when we caught an interrupt and
           * we need to stop the pipeline. */
          GST_INFO ("Interrupt: Stopping pipeline ...");
          /* return TRUE when we caught an interrupt */
          res = TRUE;
          goto exit;
        }
      }
      default:
        /* just be quiet by default */
        break;
    }
    if (message)
      gst_message_unref (message);
  }
  g_assert_not_reached ();

exit:
  {
    if (message)
      gst_message_unref (message);
    gst_object_unref (bus);
    return res;
  }
}

#if 0
int
main (int argc, char *argv[])
{
  gboolean res;
  GError *error = NULL;
  GstElement *pipeline;
  GstElement *e;

  if (!g_thread_supported ())
    g_thread_init (NULL);

  gst_init (NULL, NULL);

  pipeline = (GstElement *)
    gst_parse_launch ("filesrc ! schrodec ! schroenc ! filesink", &error);

  if (!pipeline) {
    if (error) {
      GST_INFO("ERROR: pipeline could not be constructed: %s.",
          GST_STR_NULL (error->message));
      g_error_free (error);
    } else {
      GST_INFO("ERROR: pipeline could not be constructed.");
    }
    return 1;
  } else if (error) {
    GST_INFO("WARNING: erroneous pipeline: %s",
        GST_STR_NULL (error->message));
    g_error_free (error);
    return 1;
  }

  e = gst_bin_get_by_name (GST_BIN(pipeline), "filesrc0");
  g_assert(e != NULL);
  g_object_set (e, "location", "/home/ds/media/masters/ants-2.HD720-master.drc",
      NULL);

  e = gst_bin_get_by_name (GST_BIN(pipeline), "filesink0");
  g_assert(e != NULL);
  g_object_set (e, "location", "test.drc", NULL);


  res = run_pipeline (pipeline);

  return res;
}
#endif

gboolean
run_pipeline (GstElement *pipeline)
{
  gboolean res = TRUE;
  GstState state, pending;
  GstStateChangeReturn ret;
  gboolean caught_error = FALSE;

  GST_INFO("Setting pipeline to PAUSED ...");
  ret = gst_element_set_state (pipeline, GST_STATE_PAUSED);

  switch (ret) {
    case GST_STATE_CHANGE_FAILURE:
      GST_INFO("ERROR: Pipeline doesn't want to pause.");
      res = FALSE;
      event_loop (pipeline, FALSE, GST_STATE_VOID_PENDING);
      goto end;
    case GST_STATE_CHANGE_NO_PREROLL:
      GST_INFO("Pipeline is live and does not need PREROLL ...");
      is_live = TRUE;
      break;
    case GST_STATE_CHANGE_ASYNC:
      GST_INFO("Pipeline is PREROLLING ...");
      caught_error = event_loop (pipeline, TRUE, GST_STATE_PAUSED);
      if (caught_error) {
        GST_INFO("ERROR: pipeline doesn't want to preroll.");
        goto end;
      }
      state = GST_STATE_PAUSED;
      /* fallthrough */
    case GST_STATE_CHANGE_SUCCESS:
      GST_INFO("Pipeline is PREROLLED ...");
      break;
  }

  caught_error = event_loop (pipeline, FALSE, GST_STATE_PLAYING);

  if (caught_error) {
    GST_INFO("ERROR: pipeline doesn't want to preroll.");
  } else {
    GST_INFO("Setting pipeline to PLAYING ...");
    if (gst_element_set_state (pipeline,
            GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
      GstMessage *err_msg;
      GstBus *bus;

      GST_INFO("ERROR: pipeline doesn't want to play.");
      bus = gst_element_get_bus (pipeline);
      if ((err_msg = gst_bus_poll (bus, GST_MESSAGE_ERROR, 0))) {
        GError *gerror;
        gchar *debug;

        gst_message_parse_error (err_msg, &gerror, &debug);
        gst_object_default_error (GST_MESSAGE_SRC (err_msg), gerror, debug);
        gst_message_unref (err_msg);
        g_error_free (gerror);
        g_free (debug);
      }
      gst_object_unref (bus);
      res = FALSE;
      goto end;
    }

    caught_error = event_loop (pipeline, TRUE, GST_STATE_PLAYING);
  }

  /* iterate mainloop to process pending stuff */
  while (g_main_context_iteration (NULL, FALSE));

  GST_INFO("Setting pipeline to PAUSED ...");
  gst_element_set_state (pipeline, GST_STATE_PAUSED);
  if (!caught_error)
    gst_element_get_state (pipeline, &state, &pending, GST_CLOCK_TIME_NONE);
  GST_INFO("Setting pipeline to READY ...");
  gst_element_set_state (pipeline, GST_STATE_READY);
  gst_element_get_state (pipeline, &state, &pending, GST_CLOCK_TIME_NONE);

end:
  GST_INFO("Setting pipeline to NULL ...");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_element_get_state (pipeline, &state, &pending, GST_CLOCK_TIME_NONE);

  return res;
}

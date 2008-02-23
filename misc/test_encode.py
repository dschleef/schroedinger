#!/usr/bin/env python

import sys
import gobject
import pygst
pygst.require('0.10')
import gst


class EncodeTest:
    def __init__(self):
        self.bin = gst.parse_launch('filesrc ! schrodec ! schroenc ! filesink');
        self.filesrc = self.bin.get_by_name('filesrc0');
        self.schrodec = self.bin.get_by_name('schrodec0');
        self.schroenc = self.bin.get_by_name('schroenc0');
        self.filesink = self.bin.get_by_name('filesink0');

        bus = self.bin.get_bus();
        bus.enable_sync_message_emission();
        bus.add_signal_watch();
        bus.connect('sync-message::element', self.on_sync_message);
        bus.connect('message', self.on_message);

    def set(self, name, value):
        self.schroenc.set_property(name, value);

    def set_input_file(self, input_file):
        self.filesrc.set_property('location', input_file);

    def set_output_file(self, output_file):
        self.filesink.set_property('location', output_file);

    def start(self):
        self.bin.set_state(gst.STATE_PLAYING);

    def on_sync_message(self, bus, message):
        t = message.type;
        #if message.structure:
        #    print "Sync Message: %d %s" % (t, message.structure.to_string());
        #else:
        #    print "Sync Message: %d" % t;

    def on_message(self, bus, message):
        t = message.type;
        #if message.structure:
        #    print "Message: %d %s" % (t, message.structure.to_string());
        #else:
        #    print "Message: %d" % t;
        if t == gst.MESSAGE_ERROR:
            err, debug = message.parse_error()
            print "Error: %s" % err, debug
            self.bin.set_state (gst.STATE_NULL);
        elif t == gst.MESSAGE_EOS:
            print "EOS"
            self.bin.set_state (gst.STATE_NULL);
            self.mainloop.quit();

    def stop(self):
        self.bin.set_state (gst.STATE_NULL);

    def go(self):
        self.mainloop = gobject.MainLoop();
        self.start();
        try:
            self.mainloop.run();
        except KeyboardInterrupt:
            pass
        self.stop ();


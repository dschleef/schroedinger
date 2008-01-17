
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <schroedinger/schroutils.h>
#include <schroedinger/schrodomain.h>
#include <schroedinger/schrodebug.h>
#include <stdlib.h>


/* SchroMemoryDomain */

/* hack so we don't have to fix the rest of the library right now */
static SchroMemoryDomain *
get_default_domain (void)
{
  static SchroMemoryDomain *default_domain;
  if (default_domain == NULL) {
    default_domain = schro_memory_domain_new();
  }
  return default_domain;
}

SchroMemoryDomain *
schro_memory_domain_new (void)
{
  pthread_mutexattr_t mutexattr;
  SchroMemoryDomain *domain;

  domain = schro_malloc0 (sizeof(SchroMemoryDomain));

  pthread_mutexattr_init (&mutexattr);
  pthread_mutex_init (&domain->mutex, &mutexattr);
  pthread_mutexattr_destroy (&mutexattr);

  return domain;
}

void
schro_memory_domain_free (SchroMemoryDomain *domain)
{
  int i;

  if (domain == NULL) domain = get_default_domain();

  for(i=0;i<SCHRO_MEMORY_DOMAIN_SLOTS;i++){
    if (domain->slots[i].flags & SCHRO_MEMORY_DOMAIN_SLOT_ALLOCATED) {
      free (domain->slots[i].ptr);
    }
  }

  schro_free (domain);
}

void *
schro_memory_domain_alloc (SchroMemoryDomain *domain, int size)
{
  int i;
  void *ptr;

  if (domain == NULL) domain = get_default_domain();

  SCHRO_DEBUG("alloc %d", size);

  pthread_mutex_lock (&domain->mutex);
  for(i=0;i<SCHRO_MEMORY_DOMAIN_SLOTS;i++){
    if (!(domain->slots[i].flags & SCHRO_MEMORY_DOMAIN_SLOT_ALLOCATED)) {
      continue;
    }
    if (domain->slots[i].flags & SCHRO_MEMORY_DOMAIN_SLOT_IN_USE) {
      continue;
    }
    if (domain->slots[i].size == size) {
      domain->slots[i].flags |= SCHRO_MEMORY_DOMAIN_SLOT_IN_USE;
      SCHRO_DEBUG("got %p", domain->slots[i].ptr);
      ptr = domain->slots[i].ptr;
      goto done;
    }
  }

  for(i=0;i<SCHRO_MEMORY_DOMAIN_SLOTS;i++){
    if (domain->slots[i].flags & SCHRO_MEMORY_DOMAIN_SLOT_ALLOCATED) {
      continue;
    }

    domain->slots[i].flags |= SCHRO_MEMORY_DOMAIN_SLOT_ALLOCATED;
    domain->slots[i].flags |= SCHRO_MEMORY_DOMAIN_SLOT_IN_USE;
    domain->slots[i].size = size;
    domain->slots[i].ptr = malloc (size);

    SCHRO_DEBUG("created %p", domain->slots[i].ptr);
    ptr = domain->slots[i].ptr;
    goto done;
  }

  SCHRO_ASSERT(0);
done:
  pthread_mutex_unlock (&domain->mutex);
  return ptr;
}

void
schro_memory_domain_memfree (SchroMemoryDomain *domain, void *ptr)
{
  int i;

  if (domain == NULL) domain = get_default_domain();

  SCHRO_DEBUG("free %p", ptr);

  pthread_mutex_lock (&domain->mutex);
  for(i=0;i<SCHRO_MEMORY_DOMAIN_SLOTS;i++){
    if (!(domain->slots[i].flags & SCHRO_MEMORY_DOMAIN_SLOT_ALLOCATED)) {
      continue;
    }
    if (!(domain->slots[i].flags & SCHRO_MEMORY_DOMAIN_SLOT_IN_USE)) {
      continue;
    }
    if (domain->slots[i].ptr == ptr) {
      domain->slots[i].flags &= (~SCHRO_MEMORY_DOMAIN_SLOT_IN_USE);
      pthread_mutex_unlock (&domain->mutex);
      return;
    }
  }
  pthread_mutex_unlock (&domain->mutex);

  SCHRO_ASSERT(0);
}


// Keep one device implementation of the whole-body search in the linked image.
#ifndef CUDARL_POND_GEOMETRY_RDC
#error "Compile the shared pond inspection with CUDARL_POND_GEOMETRY_RDC"
#endif
#define CUDARL_DEFINE_POND_INSPECT
#ifdef CUDARL_POND_GEOMETRY_MODEL_NAMESPACE
#define cudarl CUDARL_POND_GEOMETRY_MODEL_NAMESPACE
#endif
#include "cuda/pond_math_profile.h"

#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cuda_utils::cuda_utils" for configuration ""
set_property(TARGET cuda_utils::cuda_utils APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(cuda_utils::cuda_utils PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libcuda_utils.so"
  IMPORTED_SONAME_NOCONFIG "libcuda_utils.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS cuda_utils::cuda_utils )
list(APPEND _IMPORT_CHECK_FILES_FOR_cuda_utils::cuda_utils "${_IMPORT_PREFIX}/lib/libcuda_utils.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

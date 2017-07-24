def __bootstrap():
   global __bootstrap, __loader, __file
   import sys, pkg_resources, imp
   __file = pkg_resources.resource_filename(__name__,'MapUtils.so')
   __loader = None; del __bootstrap, __loader
   imp.load_dynamic(__name__,__file)
__bootstrap()

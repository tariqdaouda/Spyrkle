
def get_unique_filename(filename):
    import os

    def _get_name(basename, extension) :
        if len(extension) == 0:
            return basename
        return basename + "." + extension

    basename, extension = os.path.splitext(filename)

    new_basename = basename
    i = 1
    while os.path.exists( _get_name(new_basename, extension)) :
        new_basename = basename + "_%s" % i
        i += 1

    return _get_name(new_basename, extension)

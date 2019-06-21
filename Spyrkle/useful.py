
def get_unique_filename(filename):
    '''Creates a unique filename if tested filename exists'''
    import os

    sfilename = filename.split(".")
    extension = sfilename[-1]
    basename = '.'.join(sfilename[:-1])

    new_basename = basename
    i = 1
    while os.path.exists( new_basename + "." + extension) :
        new_basename = basename + "_%s" % i
        i += 1
    return new_basename + "." + extension
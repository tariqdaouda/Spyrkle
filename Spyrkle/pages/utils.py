import useful as US

class Image(object):
    """docstring for Image"""
    def __init__(self, url_or_plot, extension=".png"):
        super(Image, self).__init__()
        self.url_or_plot = url_or_plot
        self.extension = extension
        self.supported_libraries = {
            "savefig": ("seaborn", "matplotlib"),
            "save": ("altair", "ggplot2")
        }

    def get_src(self, save_folder, name=None, *args, **kwargs) :
        import os
        import uuid

        if name is None :
            new_name = str(uuid.uuid4())
        else :
            new_name = name

        filename = US.get_unique_filename( os.path.join(save_folder, new_name) + self.extension )

        if type(name) is str and ( name[:4] == "http" or name[:3] == "ftp" ) :
            #save an url
            import urllib.request
            urllib.request.urlretrieve(url, filename)  
        elif :
            saved = False
            for fct_name in self.supported_libraries :
                try:
                    getattr(self.url_or_plot, fct_name)(filename, *args, **kwargs)
                    saved = True
                    break
                except AttributeError as e:
                    pass
            if not saved :
                str_suppored = ['urls']
                for v in self.supported_libraries.values() :
                    str_suppored.extend(v)
                str_suppored = ', '.join(str_suppored)
                raise ValueError("Unknown image library, supported formats: %s" % str_suppored)

        return filename
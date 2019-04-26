import Spyrkle.useful as US

class Image(object):
    """docstring for Image"""
    def __init__(self, notebook, url_or_plot, extension=".png"):
        super(Image, self).__init__()
        self.name = notebook.name
        self.url_or_plot = url_or_plot
        self.extension = extension
        self.supported_libraries = {
            "savefig": ("seaborn", "matplotlib"),
            "save": ("altair")
        }

    def get_src(self, save_folder = "temp_figs", name=None, *args, **kwargs) :
        import os
        import uuid
        import shutil

        folder = os.path.join(".", self.name.replace(" ", "_").lower(), save_folder)

        if not os.path.isdir(folder) :
            os.makedirs(folder)

        if name is None :
            new_name = str(uuid.uuid4())
        else :
            new_name = name

        filename = US.get_unique_filename( os.path.join(folder, new_name) + self.extension )

        if type(self.url_or_plot) is str and ( self.url_or_plot[:4] == "http" or self.url_or_plot[:3] == "ftp" ) :
            #save an url
            import urllib.request
            urllib.request.urlretrieve(self.url_or_plot, filename)  
        else :
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
        #
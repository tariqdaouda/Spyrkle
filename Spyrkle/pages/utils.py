import Spyrkle.useful as US
from PIL import Image

# class Abtract_SaveWrapper(object):
#     '''
#     Class to store objects and their associated save information
#     obj: object to be saved
#     filename: name of output file
#     save_function: function necessary to save the object
#     '''
#     def __init__(self, obj, filename, save_fuction):
#         super(SaveWrapper, self).__init__()
#         self.obj = obj
#         self.filename = filename
#         self.save_fuction = save_fuction
    
#     def save(self, folder_filepath) :
#         raise NotImplemented("Should be implemented in child")

# class URLSaver(Abtract_SaveWrapper):
#     '''
#     Class specifically to save images/objects from a URL
#     url: url of the source object
#     filename: Name of the file to use when storing locally
#     '''
#     def __init__(self, url, filename):
#         super(URLSaver, self).__init__(url, filename, urllib.request.urlretrieve)
    
#     def save(self, folder_filepath):
#         import os
#         self.save_fuction(self.obj, os.path.join(folder_filepath, self.filename))
        
class Figure(object):
    '''
    Class for images/figures to be included in the notebook
    url_or_plot : either the url to an image or a python-produced plot
    extension: file format for output image, default is .png
    supported_libraries: dict, keys are names of save function, values are associated libraries
    '''
    def __init__(self, url_or_plot, name=None, extension=".png"):
        super(Figure, self).__init__()
        import uuid
        import io

        if name is None:
            self.name = str(uuid.uuid4())
        else :
            self.name = name

        self.url_or_plot = url_or_plot
        self.extension = extension
        self.supported_libraries = {
            "savefig": ("seaborn", "matplotlib"),
            "save": ("altair", "PIL")
        }
        
        self.image = None
        self.buffer = io.BytesIO()
        self._save_memory(url_or_plot)

    def _save_memory(self, url_or_plot) :
        # import shutil

        if type(self.url_or_plot) is str and ( self.url_or_plot[:4] == "http" or self.url_or_plot[:3] == "ftp" ) :
            #save an url
            import urllib.request
            # urllib.request.urlretrieve(self.url_or_plot, filename)  
            urllib.request.urlretrieve(self.url_or_plot, self.buffer)  
        else :
            saved = False
            for fct_name in self.supported_libraries :
                try:
                    # getattr(self.url_or_plot, fct_name)(filename, *args, **kwargs)
                    getattr(self.url_or_plot, fct_name)(self.buffer)
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

        # plt.savefig(buf, format='png')
        self.buffer.seek(0)
        self.image = Image.open(self.buffer)
        # self.buffer.close()

    def get_src(self, save_folder, name=None):
        import os
        import uuid

        if name is None :
            new_name = str(uuid.uuid4())
        else :
            new_name = name

        filename = US.get_unique_filename( os.path.join(save_folder, new_name) + self.extension )
        self.image.save(filename)
        return filename

class Text(object):
    """docstring for Text"""
    def __init__(self, txtfile):
        #super(Abstract_Page, self).__init__()
        self.txtfile = txtfile
        # self.extension = extension
        # self.file_path = file_path
        self.content = {}
        self.title = ""
        self.abstract = ""
        self.body = ""
        # self.supported_libraries = {}
    
    # run set_content() to get a dictionary of title, abstract, and body
    def set_content(self, title = True, abstract = True, body = True) : # Need to create optionality for end of file
        import os
        with open(self.txtfile, "rt") as myfile:
            contents = myfile.read()
        
        if title:
            title_start = contents.find("Title:")+len("Title:")
            if contents[title_start] == ' ':
                title_start += 1
            title_end = contents.find("Abstract:")-1
        
        if abstract:
            abstract_start = contents.find("Abstract:")+len("Abstract:")
            if contents[abstract_start] == ' ':
                abstract_start += 1
            abstract_end = contents.find("Body:")-1
        
        if body:
            body_start = contents.find("Body:")+len("Body:")
            if contents[body_start] == ' ':
                body_start += 1
        
        self.content["Title"] = contents[title_start:title_end]
        self.content["Abstract"] = contents[abstract_start:abstract_end]
        self.content["Body"] = contents[body_start:]
    
    # extracting title, abstract, and body from the dictionary 'self.content'
    def get_title(self):
        self.title = self.content["Title"]
        return(self.title)
    def get_abstract(self):
        self.abstract = self.content["Abstract"]
        return(self.abstract)
    def get_body(self):
        self.body = self.content["Body"]
        return(self.body)

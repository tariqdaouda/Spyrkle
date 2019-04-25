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











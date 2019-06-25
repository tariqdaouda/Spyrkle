
from collections import OrderedDict
from . import useful as US

class Notebook(object):
    '''
    Contained within Notebook is the ability to create a new notebook, add pages, save HTML output
     and render within a jupyter interface
    
    name: string, name of the notebook
    pages: stores pages of the notebook
    lib_folder: string, name of directory where necessary libraries will be stored
    static_folder: string, name of directory where static js/css files are stored
    registered_folders: dict, keys are all necessary filepaths, values are corresponding objects to be stored in filepath
    dirname: string: filepath where all notebook contents will be stored
    web_libs_dir: string, filepath for necessary libraries
     '''
    def __init__(self, name, lib_folder="libs", static_folder="static"):

        super(Notebook, self).__init__()
        import os
        import sys
        import inspect
        # Define the name, pages, libs, etc. of the notebook
        self.name = name
        self.pages = OrderedDict()
        #self.figures = []
        self.lib_folder = lib_folder
        self.static_folder = static_folder
        self.registered_folders = {}
        self.dirname = os.path.dirname(inspect.getfile(sys.modules[__name__]))
        self.web_libs_dir = os.path.join(self.dirname, "static/libs")

    def add_page(self, page) :
        '''Adds a page to the notebook'''
        self.pages[page.name] = page # page.name is notes.name, which calls the name of notes given when notes object was created

    def register_folder(self, filepath, overwrite) :
        '''Add folder to dict of folders.  Creates new key/value pair'''
        self.registered_folders[filepath] = {'flags': {'overwrite': True}, 'objects': []}

    def add_to_registered_folder(self, filepath, obj) :
        '''Add object to an already registered folder'''
        self.registered_folders[filepath].append(obj)
    
    def _create_registered_folders(self, parent_folder) :
        '''Create all registered folders in order of their location within the output folder tree '''
        import os
        import shutil

        for fp, data in sorted(self.registered_folders.items(), key = lambda file: file[0].count("/")) :
            foldername = os.path.join(parent_folder, fp)
            try:
                os.mkdir(foldername)
            except FileExistsError as e:
                if not data['flags']['overwrite'] :
                    print("Warning: Folder %s already exists" % foldername)
                else :
                    shutil.rmtree(foldername)
                    os.mkdir(foldername)

    def get_html(self, jupyter = False) :
        '''
        returns the html of the output notebook
        jupyter: boolean, if true, adjusts filepaths for proper output in a jupyter ipython notebook, will be called if "view" is used
        '''

        # Need to change the path pointed to if being rendered in a Jupyter notebook
        if jupyter:
            main_dir = self.name.replace(" ", "_").lower()
        else :
            main_dir = "."

        switch_html = []
        switch_menu = []

        # For each page, create the proper HTML/css
        i = 0
        css_links = []
        for name, page in self.pages.items() :
            page_id = 'page-%s' % i
            switch_menu.append( """<spyrkle-page-selector onclick="toggle_page('{pid}')" class='uk-button uk-button-default' id='{pid}-selector'>{name}</spyrkle-page-selector>""".format(name=page.name, pid=page_id) )
            switch_html.append( "<spyrkle-page id='{pid}'>\n{html}\n</spyrkle-page>".format(pid=page_id, html=page.get_html()) )
            # Add appropriate css if exists for a page
            if page.has_css() :
                css_links.append('<link rel="stylesheet" href="{main_dir}/static/css/{name}.css" />'.format(name=name, main_dir = main_dir))
            i += 1

        # Define the head HTML, name refers to name of notebook
        head = """
        <head>
            <!doctype html>
            <meta charset="utf-8">
            <title>{name}</title>
        </head>""".format(name = self.name)
        
        # set up the javascript
        js ="""
            <!-- JQUERY -->
            <script src="{main_dir}/{libs_dir}/jquery/js/jquery-3.4.0.min.js"></script>
            
            <!-- UIkit CSS -->
            <link rel="stylesheet" href="{main_dir}/{libs_dir}/uikit-3.0.3/css/uikit.css" />

            <!-- UIkit JS -->
            <script src="{main_dir}/{libs_dir}/uikit-3.0.3/js/uikit.min.js"></script>
            <script src="{main_dir}/{libs_dir}/uikit-3.0.3/js/uikit-icons.min.js"></script>

            <!-- Spyrkle Pages CSS -->
            {pages_css}

            <!-- SPYRKLE JS-->
            <script src="{main_dir}/{libs_dir}/spyrkle/js/spyrkle.js"></script>

            
        """.format(pages_css = "\n".join(css_links), libs_dir=self.lib_folder, main_dir = main_dir)
        
        # Define header, page switches, body of pages
        header="<h1 class='uk-header-primary uk-margin uk-text-center'>{name}</h1>".format(name=self.name)
        switcher='<div class="uk-button-group uk-margin">{menu}</div>'.format(menu=''.join(switch_menu)) 
        switcher_html='<div class="uk-container">{data}</div>'.format(data=''.join(switch_html)) 
        body = """{js}<body onload="toggle_page('page-0')" class='uk-container'>{header}\n{switcher}\n{switcher_html}\n</body>""".format(header=header, switcher=switcher, switcher_html=switcher_html, js=js)
        
        # Add a footer with a small note about the application    
        footer = "<footer class='uk-text-meta uk-text-center uk-margin'><p class='uk-text-meta'>Generetad by Spyrkle, static documentation for your glorious pythonic work</p></footer>"
        
        # Return all of the formated HTML/CSS/js together in correct order
        return '\n'.join((head, body, footer))

    def save(self, folder = ".", overwrite = False) :
        '''
        Saves output HTML, necessary libraries for a notebook into a given directory
        folder: string, filepath where the notebook should be saved, default is current directory
        overwrite: boolean, if true, will overwrite already existing notebook files with same name. If false, will create notebook under new name if current notebook name exists
        '''
        import os
        import shutil

        def _populate_folder(folder_fp, objects_fp) :
            for obj in objects_fp :
                shutil.copy(objects_fp, folder_fp)

        foldername = os.path.join(folder, self.name.replace(" ", "_").lower())


        new_foldername = foldername
        if not overwrite :
            new_foldername = US.get_unique_filename(new_foldername)

        # Create paths for static files
        static_folder = os.path.join(self.static_folder)
        css_folder = os.path.join(self.static_folder, "css")
        js_folder = os.path.join(self.static_folder, "js")

        # Create a path for a library folder, default name of lib_folder is "libs"
        libs_folder = os.path.join(self.lib_folder)

        # Create a figs folder
        # figs_folder = os.path.join(new_foldername, self.figs_folder)

        # Create the folder
        self.register_folder('', overwrite=False)
        self.register_folder(static_folder, overwrite=False)
        self.register_folder(css_folder, overwrite=False)
        self.register_folder(js_folder, overwrite=False)
        self.register_folder(libs_folder, overwrite=True)
        self._create_registered_folders(parent_folder = new_foldername)
        
        # Copy the library directory to the libs folder
        for libs in os.listdir(os.path.join(self.web_libs_dir)) :
            shutil.copytree(os.path.join(self.web_libs_dir, libs), os.path.join(new_foldername, libs_folder, libs))

        # For every page, write proper css files
        for name, page in self.pages.items() :
            fn = os.path.join(new_foldername, css_folder, "%s.css" % name)
            f = open(fn, "w")
            f.write(page.get_css())
            f.close()

        # Write the HTML page for the notebook
        fn = os.path.join(new_foldername, "index.html") # create a file index.html inside folder new_foldername, a name that is given when running notebook.Note("notebook name")
        f = open(fn, "w") # open index.html
        f.write(self.get_html()) # run notebook.py's get_html
        f.close()
    
    def view(self):
        '''Functionality to view notebook output in an ipython Jupyter notebook'''
        import re
        import os
        from IPython.core.display import display, HTML
        # Save reference libraries/output html
        self.save(overwrite = True)
        
        # Display the HTML in the Jupyter notebook
        ret = self.get_html(jupyter = True)

        # Change path of figures        
        # ret = re.sub(r"(.?)(.?)figs(.?)", "/".join([self.name.replace(" ", "_").lower(), "figs", ""]), ret)
        return display(HTML(ret))
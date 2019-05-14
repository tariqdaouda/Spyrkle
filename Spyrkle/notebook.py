
from collections import OrderedDict
from . import useful as US

class Notebook(object):
    '''Contained within Notebook is the ability to create a new notebook, add pages, save HTML output
     and render within a jupyter interface'''
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

    # Function to add a page to a notebook, takes in something of class "page" which is in core_pages 
    def add_page(self, page) :
        '''Adds a page to the notebook'''
        self.pages[page.name] = page

    def register_folder(self, filepath, overwrite) :
        self.registered_folders[filepath] = {'flags': {'overwrite': True}, 'objects': []}

    def add_to_registered_folder(self, filepath, obj) :
        self.registered_folders[filepath].append(obj)
    
    def _create_registered_folders(self, parent_folder) :
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

    # def _folder_tree(self) :
    #     tree = []
    #     # Take in registered folders
    #     for path in self.registered_folders :
    #         # Get their "level"
            

    #         # Sort then by thier level


    # Function to get html of notebook
    def get_html(self, jupyter = False) :
        '''Get the html of notebook'''

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
        '''Saves output HTML, necessary libraries for a notebook into a given directory'''
        import os
        import shutil

        def _populate_folder(folder_fp, objects_fp) :
            for obj in objects_fp :
                shutil.copy(objects_fp, folder_fp)
                # os.remove(os.path.join(temp, obj))

        # Create folder name/path based on notebook name
        foldername = os.path.join(folder, self.name.replace(" ", "_").lower())

        new_foldername = foldername
        # If overwrite is False and the folder already exists, make a new unique folder
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
        print(self.registered_folders)
        print(new_foldername)
        self._create_registered_folders(parent_folder = new_foldername)
        
            # for obj in self.registered_folders[fp] :
            #     obj.save(full_fp)
            
            # _populate_folder(new_foldername, wrapped_objects)

        #_create_folder( os.path.join(new_foldername, self.lib_folder) )

        # # If the libs folder already exists, remove it


        # If the figs folder already exists, remove all contents
        # if os.path.isdir(figs_folder) :
        #     for fig in os.listdir(figs_folder):
        #         os.remove(os.path.join(figs_folder, fig))
        
        # Copy the library directory to the libs folder
        shutil.copytree(os.path.join(self.web_libs_dir), os.path.join(new_foldername, libs_folder))

        # For every page, write proper css files
        for name, page in self.pages.items() :
            fn = os.path.join(css_folder, "%s.css" % name)
            f = open(fn, "w")
            f.write(page.get_css())
            f.close()

        # Write the HTML page for the notebook
        fn = os.path.join(new_foldername, "index.html")
        f = open(fn, "w")
        f.write(self.get_html())
        f.close()
    
    def view(self):
        print(self.dirname)
        import re
        import os
        """Returns an interactive visualization for JuPyter"""
        from IPython.core.display import display, HTML
        # Save reference libraries/output html
        self.save(overwrite = True)
        
        # Display the HTML in the Jupyter notebook
        ret = self.get_html(jupyter = True)

        # Change path of figures        
        ret = re.sub(r"(.?)(.?)figs(.?)", "/".join([self.name.replace(" ", "_").lower(), "figs", ""]), ret)
        #print(ret)
        return display(HTML(ret))
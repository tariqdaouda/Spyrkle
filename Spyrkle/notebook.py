
from collections import OrderedDict
from . import useful as US
import uuid

class AnchorNav(object):
    """Represent the sticky nav bar on top of a page"""
    def __init__(self, page):
        super(AnchorNav, self).__init__()
        # self.page = page
        self.page_id = page = "top-%s" % page.name
        self.sections = {}
        self.navs=OrderedDict()

    def add(self, sect, anchor):
        """Andd a section to the menu. To create create a dropdown menu use <main-section>.<sub-section> as anchor"""
        unique_key = "%s_%d" % (self.page_id, len(self.sections))
        self.sections[sect] = unique_key

        sanchor = anchor.split(">")
        # if len(sanchor) > 2:
            # raise ValueError("Anchor can only have one sub section: %s has %d" % (anchor, len(sanchor)))

        if sanchor[0] not in self.navs:
            self.navs[sanchor[0]] = {
                "nav": anchor,
                "id": unique_key,
                "subs": []
            }
        
        if len(sanchor) > 1 :
            nav = {
                "nav": sanchor[1],
                "id": unique_key
            }
            self.navs[sanchor[0]]["subs"].append(nav)
  
    def get_html(self):
        anchors = []
        for anch in self.navs.values():
            if len(anch["subs"]) == 0 :
                html = '<li><a href="#%s">%s</a></li>' % (anch["id"], anch["nav"])
            else:
                subs = ['<li><a href="#%s">%s</a></li>' % (sub_anch["id"], sub_anch["nav"]) for sub_anch in anch["subs"] ]
                html = """
                <li>
                    <a href="#{parent_id}">{parent}</a>
                    <div class="uk-navbar-dropdown">
                        <ul class="uk-nav uk-navbar-dropdown-nav">
                            {items}
                        </ul>
                    </div>
                </li>
                """.format(parent_id =anch["id"], parent=anch["nav"], items='\n'.join(subs) )
            anchors.append(html)

        anchor_nav= """
        <span id={pageid}></span>
        <div class="uk-card uk-card-default uk-card-body" style="z-index: 980;" uk-sticky="offset: 100">
            <nav class="uk-navbar-container" uk-navbar>
                <div class="uk-navbar-left">
                    <ul class="uk-navbar-nav">
                        {anchors}
                    </ul>
                </div>
                <div class="uk-navbar-right">
                    <ul class="uk-navbar-nav">
                        <li><a href="#{pageid}">^Top</a></li>
                    </ul>
                </div>
            </nav>
        </div>
        """.format( anchors='\n'.join(anchors), pageid=self.page_id)
        return anchor_nav

    def is_empty(self):
        return len(self.sections) == 0

    def __getitem__(self, sect):
        return self.sections[sect]

    def __contains__(self, sect):
        return sect in self.sections

class Page(object):
    """A spyrkle Page"""
    def __init__(self, notebook, name):
        super(Page, self).__init__()
        self.name = name
        self.notebook = notebook
        self.sections = OrderedDict()
        self.folder = None
        self.anchors = AnchorNav(self)

    def _set_folder(self, fol):
        self.folder = fol

    def add_section(self, section, anchor:str=None):
        """Add a section to the page.
        If anchor is not None, it will be added to a sticky nav bar on toop of the page.
        To create create a dropdown menu use <main-section>'>'<sub-section> as anchor
        """
        self.sections[section.name] = section
        if anchor is not None: self.anchors.add(section, anchor)
        section._set_page(self)

    def has_css(self) :
        '''Indicates if page has associated css'''
        for sect in self.sections.values():
             if len(sect.css_rules) > 0 :
                return True
        return False

    def get_css(self) :
        '''Returns list of css for page'''
        res = []
        for sect in self.sections.values():
            res.append( sect.get_css() ) ;
        return '\n'.join(res)

    def get_html(self) :
        '''Gets html for page'''
        res = []
        for sect in self.sections.values():
            sect_html = sect.get_html()
            if sect in self.anchors:
                sect_html = "<span id='%s'></span>%s" % (self.anchors[sect], sect_html)
            res.append( sect_html ) ;
        html =  '\n'.join(res)

        if not self.anchors.is_empty(): html = self.anchors.get_html() + html 
        return html

    def __getitem__(self, name):
        return self.sections[name]

    # def __del__(self, name):
        # del self.sections[name]

class Abstract_Section(object):
    '''
    Abstract section that sections will inherit from
    name : string, name of section
    static_urls : set of urls for static files
    lib_urls : set of urls for libraries
    css_rules : dict, containing info about css for a section
    '''
    def __init__(self, name = None):
        super(Abstract_Section, self).__init__()
        # self.notebook = notebook
        if name is None :
            self.name = "%s-%s" % (self.__class__.__name__, str(uuid.uuid4()) )
        else:
            self.name = name

        self.static_urls = set()
        self.libs_urls = set()
        
        # self.notebook.add_section(self) # run 'BOOK.add_section', a method in object BOOK, create a sub-section under notebook BOOK, putting in itself (object notes) as an argument.
        self.css_rules = {}
        # self.js_scripts = {}
        self.reset_css()
        self.page = None

    def _set_page(self, page):
        self.page = page

    def has_css(self) :
        '''Indicates if a section has associated css'''
        return len(self.css_rules) > 0

    def register_static(self, url) :
        '''Add url to static url set'''
        self.static_urls.add(url) 
    
    def register_lib(self, url) :
        '''Add library url to lib set'''
        self.lib_urls.add(url)

    def clear_css(self) :
        '''Clear all css of a section, create empty dict'''
        self.css_rules = {}

    def set_css_rule(self, name, lst) :
        '''
        Set css for a section
        name: string, name of a section
        lst: list of strings defining css
        '''
        self.css_rules[name] = lst

    def reset_css(self) :
        '''Reset css'''
        pass

    def get_css(self) :
        '''Returns list of css for section'''
        res = []
        for n, rules in self.css_rules.items() :
            str_rules = '; '.join(rules)
            res.append( "%s {%s} " % (n, str_rules) ) ;
        return '\n'.join(res)

    def get_html(self) :
        '''Gets html for a section'''
        raise NotImplemented("Must be implemented in child")

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
    def __init__(self, name, lib_folder="libs", static_folder="static", save_folder="."):

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
        self.final_save_folder = None
        self.save_folder = save_folder

    def remove_self_url_root(self, url):
        return "." + url[len(self.final_save_folder):]

    def new_page(self, name):
        """create a new page"""
        page = Page(self, name)
        self.pages[page.name] = page
        return page

    def add_page(self, section_or_name) :
        '''Adds a page to the notebook'''
        if isinstance(section_or_name, Abstract_Section):
            page = Page(self, section_or_name.name)
            page.add_section(section_or_name)
        else :
            page = Page(self, section_or_name)

        self.pages[page.name] = page
        return page

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
        body = """<body onload="toggle_page('page-0')" class='uk-container'>{js}{header}\n{switcher}\n{switcher_html}\n</body>""".format(header=header, switcher=switcher, switcher_html=switcher_html, js=js)
        
        # Add a footer with a small note about the application    
        footer = "<footer class='uk-text-meta uk-text-center uk-margin'><p class='uk-text-meta'>Generetad by Spyrkle, static documentation for your glorious pythonic work</p></footer>"
        
        # Return all of the formated HTML/CSS/js together in correct order
        return "<html>" + '\n'.join((head, body, footer)) + "</html>"

    def save(filename) :
        """save a pickled version of self"""
        import pickle
        with open(filename, "wb") as f :
            pickle.dump(self, f)

    def set_save_folder(self, folder):
        self.save_folder = folder

    def export(self, save_folder = None, overwrite = False, force_new=False) :
        '''
        Saves output HTML, necessary libraries for a notebook into a given directory
        save_folder: string, filepath where the notebook should be saved, default is current directory
        overwrite: boolean, if true, will overwrite already existing notebook files with same name
        on the first export. If false, will create notebook under new name if current notebook name exists.
        Has no effect for subsequent exports.
        force_new: boolean for subsequent exports. Will force the creation of a new notebook even if the notebook was previoulsy exported 
        '''
        import os
        import shutil

        def _populate_save_folder(folder_fp, objects_fp) :
            for obj in objects_fp :
                shutil.copy(objects_fp, folder_fp)

        if (self.final_save_folder is None) or (force_new) or ( (save_folder is not None) and (self.save_folder != save_folder) ) :
            if save_folder is not None :
                self.save_folder = save_folder

            self.save_folder = os.path.join(self.save_folder, self.name.replace(" ", "_").lower())

            self.final_save_folder = self.save_folder
            if not overwrite :
                self.final_save_folder = US.get_unique_filename(self.final_save_folder)

        # Create paths for static files
        static_folder = os.path.join(self.static_folder)
        css_folder = os.path.join(self.static_folder, "css")
        img_folder = os.path.join(self.static_folder, "img")
        js_folder = os.path.join(self.static_folder, "js")
        pages_folder = os.path.join(self.static_folder, "pages")

        # Create a path for a library folder, default name of lib_folder is "libs"
        libs_folder = os.path.join(self.lib_folder)

        # Create a figs folder
        # figs_folder = os.path.join(self.final_save_folder, self.figs_folder)

        # Create the folder
        self.register_folder('', overwrite=False)
        self.register_folder(static_folder, overwrite=False)
        self.register_folder(css_folder, overwrite=False)
        self.register_folder(img_folder, overwrite=False)
        self.register_folder(js_folder, overwrite=False)
        self.register_folder(pages_folder, overwrite=True)
        self.register_folder(libs_folder, overwrite=True)

        for page_name, page in self.pages.items() :
            page_folder = os.path.join(pages_folder, page_name.replace(" ", "_"))
            self.register_folder( page_folder, overwrite=True)
            page._set_folder(os.path.join(self.final_save_folder, page_folder))

        self._create_registered_folders(parent_folder = self.final_save_folder)
        
        # Copy the library directory to the libs folder
        for libs in os.listdir(os.path.join(self.web_libs_dir)) :
            shutil.copytree(os.path.join(self.web_libs_dir, libs), os.path.join(self.final_save_folder, libs_folder, libs))

        # For every page, write proper css files
        for name, page in self.pages.items() :
            fn = os.path.join(self.final_save_folder, css_folder, "%s.css" % name)
            f = open(fn, "w")
            f.write(page.get_css())
            f.close()

        # Write the HTML page for the notebook
        fn = os.path.join(self.final_save_folder, "index.html") # create a file index.html inside folder self.final_save_folder, a name that is given when running notebook.Note("notebook name")
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
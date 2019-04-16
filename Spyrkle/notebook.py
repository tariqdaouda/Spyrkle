
from collections import OrderedDict

class Notebook(object):
    """docstring for Notebook"""
    def __init__(self, name, lib_folder="libs", static_folder="static"):
        super(Notebook, self).__init__()
        import os
        import sys
        import inspect

        self.name = name
        self.pages = OrderedDict()
        self.lib_folder = lib_folder
        self.static_folder = static_folder
        
        self.dirname = os.path.dirname(inspect.getfile(sys.modules[__name__]))
        self.web_libs_dir = os.path.join(self.dirname, "static/libs")

    def add_page(self, page) :
        self.pages[page.name] = page

    def get_html(self) :
        switch_html = []
        switch_menu = []

        i = 0
        css_links = []
        for name, page in self.pages.items() :
            page_id = 'page-%s' % i
            switch_menu.append( """<spyrkle-page-selector onclick="toggle_page('{pid}')" class='uk-button uk-button-default' id='{pid}-selector'>{name}</spyrkle-page-selector>""".format(name=page.name, pid=page_id) )
            switch_html.append( "<spyrkle-page id='{pid}'>\n{html}\n</spyrkle-page>".format(pid=page_id, html=page.get_html()) )
            
            if page.has_css() :
                css_links.append('<link rel="stylesheet" href="./static/css/{name}.css" />'.format(name=name))
            i += 1

        head = """
        <head>
            <!doctype html>
            <meta charset="utf-8">
            <title>{name}</title>
        </head>""".format(name = self.name)
        
        js ="""
            <!-- JQUERY -->
            <script src="./{libs_dir}/jquery/js/jquery-3.4.0.min.js"></script>
            
            <!-- UIkit CSS -->
            <link rel="stylesheet" href="./{libs_dir}/uikit-3.0.3/css/uikit.css" />

            <!-- UIkit JS -->
            <script src="./{libs_dir}/uikit-3.0.3/js/uikit.min.js"></script>
            <script src="./{libs_dir}/uikit-3.0.3/js/uikit-icons.min.js"></script>

            <!-- Spyrkle Pages CSS -->
            {pages_css}

            <!-- SPYRKLE JS-->
            <script src="./{libs_dir}/spyrkle/js/spyrkle.js"></script>

            
        """.format(pages_css = "\n".join(css_links), libs_dir=self.lib_folder )
  
        header="<h1 class='uk-header-primary uk-margin uk-text-center'>{name}</h1>".format(name=self.name)
        switcher='<div class="uk-button-group uk-margin">{menu}</div>'.format(menu=''.join(switch_menu)) 
        switcher_html='<div class="uk-container">{data}</div>'.format(data=''.join(switch_html)) 
        body = """{js}<body onload="toggle_page('page-0')" class='uk-container'>{header}\n{switcher}\n{switcher_html}\n</body>""".format(header=header, switcher=switcher, switcher_html=switcher_html, js=js)

        footer = "<footer class='uk-text-meta uk-text-center uk-margin'><p class='uk-text-meta'>Generetad by Spyrkle, static documentation for your glorious pythonic work</p></footer>"

        return '\n'.join((head, body, footer))

    def save(self, folder = ".", overwrite = False) :
        import os
        import shutil

        def _create_folder(folder_name):
            try:
                os.mkdir(folder_name)
            except FileExistsError as e:
               print("Warning: Folder %s already exists" % folder_name)
        
        foldername = os.path.join(folder, self.name.replace(" ", "_").lower())

        new_foldername = foldername
        if not overwrite :
            i = 1
            while os.path.exists(new_foldername) :
                new_foldername = foldername + "_%s" % i
                i += 1

        static_folder = os.path.join(new_foldername, self.static_folder)
        css_folder = os.path.join(new_foldername, self.static_folder, "css")
        js_folder = os.path.join(new_foldername, self.static_folder, "js")
        libs_folder = os.path.join(new_foldername, self.lib_folder)

        _create_folder( new_foldername )
        _create_folder( static_folder )
        _create_folder( css_folder )
        _create_folder( js_folder )
        # _create_folder( os.path.join(new_foldername, self.lib_folder) )

        if os.path.isdir(libs_folder) :
            shutil.rmtree(libs_folder) 
        
        shutil.copytree(self.web_libs_dir, libs_folder)

        for name, page in self.pages.items() :
            fn = os.path.join(css_folder, "%s.css" % name)
            f = open(fn, "w")
            f.write(page.get_css())
            f.close()

        fn = os.path.join(new_foldername, "index.html")
        f = open(fn, "w")
        f.write(self.get_html())
        f.close()

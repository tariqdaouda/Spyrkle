
from collections import OrderedDict

class Notebook(object):
    """docstring for Notebook"""
    def __init__(self, name, lib_folder="libs", static_folder="static"):
        super(Notebook, self).__init__()
        self.name = name
        self.pages = OrderedDict()
        self.lib_folder = lib_folder
        self.static_folder = static_folder

    def add_page(self, page) :
        self.pages[page.name] = page

    def get_html(self) :
        switch_html = []
        switch_menu = []
        for name, page in self.pages.items() :
            switch_menu.append("<li><a href='#'>%s</a></li>" % page.name)
            switch_html.append("<li>%s</li>" % page.get_html())

        head = """<head>
            <!doctype html>
            <meta charset="utf-8">
            <!-- UIkit CSS -->
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/uikit/3.0.3/css/uikit.min.css" />

            <!-- UIkit JS -->
            <script src="https://cdnjs.cloudflare.com/ajax/libs/uikit/3.0.3/js/uikit.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/uikit/3.0.3/js/uikit-icons.min.js"></script>

            <title>Dagre D3 Demo: Tooltip on Hover</title>
        </head>"""
        
        heading="<h1 class='uk-heading-primary'>{name}</h1>".format(name=self.name)
        switcher='<ul class="uk-subnav uk-subnav-pill" uk-switcher>{menu}</ul>'.format(menu=''.join(switch_menu)) 
        switcher_html='<ul class="uk-switcher uk-margin">{data}</ul>'.format(data=''.join(switch_html)) 

        body = "<body class='uk-container'>{heading}\n{switcher}\n{switcher_html}</body>".format(heading=heading, switcher=switcher, switcher_html=switcher_html)

        footer = "<footer><p class='uk-text-meta'>Generetad by Spyrkle, static documentation for your glorious pythonic work</p></footer>"

        return '\n'.join((head, body, footer))

    def save(self, folder = ".") :
        import os

        foldername = os.path.join(folder, self.name)

        new_foldername = foldername
        i = 1
        while os.path.exists(new_foldername) :
            new_foldername = foldername + "_%s" % i
            i += 1
        os.mkdir(new_foldername)

        os.mkdir( os.path.join(new_foldername, self.lib_folder) )
        os.mkdir( os.path.join(new_foldername, self.static_folder) )

        fn = os.path.join(new_foldername, "index.html")
        f = open(fn, "w")
        f.write(self.get_html())
        f.close()

class Page(object):
    """docstring for Page"""
    def __init__(self, notebook, name):
        super(Page, self).__init__()
        self.notebook = notebook
        self.name = name
        self.static_urls = set()
        self.libs_urls = set()
        
        self.notebook.add_page(self)
    
    def register_static(self, url) :
        self.static_urls.add(url) 
    
    def register_lib(self, url) :
        self.lib_urls.add(url)

    def get_html(self) :
        pass

class Notes(Page):
    """docstring for Notes"""
    def __init__(self, notebook, name):
        super(Notes, self).__init__(notebook, name)
        self.notes_html = []

    def add_note_html(self, html, static_urls=[], lib_urls=[]) :
        self.notes_html.append(html)
        
        for e in static_urls :
            self.register_static(e)

        for e in static_libs :
            self.register_static(e)

    def add_note(self, title, body, img_src=None, add_line=True) :

        if add_line :
            import traceback
            try:
                raise TypeError("Oups!")
            except Exception as err:
                line_data = "<div class='uk-text-meta'>{filename}: {line}</div>".format(filename=traceback.extract_stack()[0].filename, line=traceback.extract_stack()[0].lineno)
        else :
            line_data = ""

        if img_src :    
            img = """
            <div class="uk-card-media-bottom">
                <img src="{img_src}" alt="">
            </div>
            """.format(img_src = img_src)
        else :
            img = ""

        html = """
        <div class="uk-card uk-card-default">
            <div class="uk-card-body">
                <h3 class="uk-card-title">{title}</h3>
                {line}
                <p>{body}</p>
            </div>
            {img}
        </div>
        """.format(line=line_data, title=title, body=body, img=img)
        self.notes_html.append(html)

    def add_bullet_points_note(self, title, points, img_src=None, add_line=True) :
        
        if add_line :
            import traceback
            try:
                raise TypeError("Oups!")
            except Exception as err:
                line_data = "<div class='uk-text-meta'>{filename}: {line}</div>".format(filename=traceback.extract_stack()[0].filename, line=traceback.extract_stack()[0].lineno)
        else :
            line_data = ""

        
        if img_src :    
            img = """
            <div class="uk-card-media-bottom">
                <img src="{img_src}" alt="">
            </div>
            """.format(img_src = img_src)
        else :
            img = ""
       
        lis = "<li>"+"</li><li>".join(points)+"</li>"

        html = """
        <div class="uk-card uk-card-default">
            <div class="uk-card-body">
                <h3 class="uk-card-title">{title}</h3>
                {line}
                <ul class="uk-list uk-list-bullet">{lis}</ul>
            </div>
            {img}
        </div>
        """.format(line=line_data, title=title, lis=lis, img=img)
        self.notes_html.append(html)

    def get_html(self) :
        html=""""
        <div class="uk-child-width-1-4@m uk-child-width-1-2@s" uk-grid="masonry: true">
        {notes}
        </div>
        """.format(notes = "\n".join(self.notes_html))

        return html

class DAG(Page):
    """docstring for DAG"""
    def __init__(self, notebook, name):
        super(DAG, self).__init__(notebook, name)
        self._init()

    def _init(self) :
        self.nodes, self.edges = set(), set()
        self.node_attributes = {}

    def set(self, nodes, edges) :
        self.nodes = set(nodes)
        self.edges = set(edges)

    def derive(self, root, get_children_fct, get_name_fct, get_attributes_fcts = None, autoincrement_names=True) :
        def _resolve_name(base_name, autoinc) :
            if not autoinc :
                return base_name

            name = base_name
            i = 1
            while name in self.nodes :
                name = base_name + "_%d" % i 
                i += 1

            return name

        def _derive(root, nodes, edges, node_attributes) :
            root_name = _resolve_name(get_name_fct(root), autoincrement_names)
            nodes.add( root_name )
            
            if get_attributes_fcts :
                for attr_name, fct in get_attributes_fcts :
                    node_attributes[attr_name] = fct(root)

            for d in get_children_fct(root) :
                if d is not root :
                    d_name = _resolve_name(get_name_fct(d), autoincrement_names)
                    edges.add((root_name, d_name))
                    _derive(d, nodes, edges, node_attributes)

        self._init()
        _derive(root, self.nodes, self.edges, self.node_attributes)


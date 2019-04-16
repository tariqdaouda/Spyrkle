
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
            <script src="../static/libs/jquery/js/jquery-3.4.0.min.js"></script>
            
            <!-- UIkit CSS -->
            <link rel="stylesheet" href="../static/libs/uikit-3.0.3/css/uikit.css" />

            <!-- UIkit JS -->
            <script src="../static/libs/uikit-3.0.3/js/uikit.min.js"></script>
            <script src="../static/libs/uikit-3.0.3/js/uikit-icons.min.js"></script>

            <!-- Spyrkle Pages CSS -->
            {pages_css}

            <!-- SPYRKLE JS-->
            <script src="../static/libs/spyrkle/js/spyrkle.js"></script>

            
        """.format(pages_css = "\n".join(css_links) )
  
        header="<h1 class='uk-header-primary'>{name}</h1>".format(name=self.name)
        switcher='<div class="uk-button-group uk-margin">{menu}</div>'.format(menu=''.join(switch_menu)) 
        switcher_html='<div class="uk-container">{data}</div>'.format(data=''.join(switch_html)) 
        body = """{js}<body onload="toggle_page('page-0')" class='uk-container'>{header}\n{switcher}\n{switcher_html}\n</body>""".format(header=header, switcher=switcher, switcher_html=switcher_html, js=js)

        footer = "<footer><p class='uk-text-meta'>Generetad by Spyrkle, static documentation for your glorious pythonic work</p></footer>"

        return '\n'.join((head, body, footer))

    def save(self, folder = ".", overwrite = False) :
        import os

        def _create_folder(folder_name):
            try:
                os.mkdir(folder_name)
            except FileExistsError as e:
               print("Warning: Folder %s already exists" % folder_name)
        
        foldername = os.path.join(folder, self.name)

        new_foldername = foldername
        if not overwrite :
            i = 1
            while os.path.exists(new_foldername) :
                new_foldername = foldername + "_%s" % i
                i += 1

        static_folder = os.path.join(new_foldername, self.static_folder)
        css_folder = os.path.join(new_foldername, self.static_folder, "css")
        js_folder = os.path.join(new_foldername, self.static_folder, "js")

        _create_folder( new_foldername )
        _create_folder( static_folder )
        _create_folder( css_folder )
        _create_folder( js_folder )
        _create_folder( os.path.join(new_foldername, self.lib_folder) )

        for name, page in self.pages.items() :
            fn = os.path.join(css_folder, "%s.css" % name)
            f = open(fn, "w")
            f.write(page.get_css())
            f.close()

        fn = os.path.join(new_foldername, "index.html")
        f = open(fn, "w")
        f.write(self.get_html())
        f.close()

class Abstract_Page(object):
    """docstring for Page"""
    def __init__(self, notebook, name):
        super(Abstract_Page, self).__init__()
        self.notebook = notebook
        self.name = name
        self.static_urls = set()
        self.libs_urls = set()
        
        self.notebook.add_page(self)
        self.css_rules = {}
        # self.js_scripts = {}
        self.reset_css()

    def has_css(self) :
        return len(self.css_rules) > 0

    def register_static(self, url) :
        self.static_urls.add(url) 
    
    def register_lib(self, url) :
        self.lib_urls.add(url)

    def clear_css(self) :
        self.css_rules = {}

    def set_css_rule(self, name, lst) :
        self.css_rules[name] = lst

    def reset_css(self) :
        pass

    def get_css(self) :
        res = []
        for n, rules in self.css_rules.items() :
            str_rules = '; '.join(rules)
            res.append( "%s {%s} " % (n, str_rules) ) ;
        return '\n'.join(res)

    def get_html(self) :
        raise NotImplemented("Must be implemented in child")

class Notes(Abstract_Page):
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

    def add_note(self, title, body, img_src=None, code=None, add_line_reference=True) :

        if add_line_reference :
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

        if code :
            lines = code.splitlines()
            l0 = None
            for l in lines :
                if len(l) != 0 :
                    l0 = l 
                    break

            if not l0 :
                code=""
            else :
                indent = len(l0) - len(l0.strip()) 
                strip_lines = [ l[indent:] for l in lines ]
                code = "<pre>%s</pre>" % '\n'.join(strip_lines)
        else :
            code = ""

        html = """
        <div class="uk-card uk-card-default">
            <div class="uk-card-body">
                <h3 class="uk-card-title">{title}</h3>
                {line}
                <p>{body}</p>
            </div>
            {code}
            {img}
        </div>
        """.format(line=line_data, title=title, body=body, code=code, img=img)
        self.notes_html.append(html)

    def add_bullet_points_note(self, title, points, img_src=None, add_line_reference=True) :
        
        if add_line_reference :
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
        html="""
        <div class="uk-child-width-1-3@m uk-child-width-1-2@s" uk-grid="masonry: true">
        {notes}
        </div>
        """.format(notes = "\n".join(self.notes_html))

        return html

class Abstract_DAG(Abstract_Page):
    """docstring for DAG"""
    def __init__(self, notebook, name):
        super(Abstract_DAG, self).__init__(notebook, name)
        self._init()

    def _init(self) :
        self.nodes, self.edges = {}, {}
        self.node_labels = set()

    def force_set(self, nodes, edges) :
        self._init()
        for n in self.nodes :
            self.node_labels.add(d)

        self.nodes = nodes
        self.edges = edges

    def parse(self, fct, *args, **kwargs) :
        self.nodes, self.edges = fct(*args, **kwargs)
    
    def resolve_node_name(self, base_name, autoinc) :
        if not autoinc :
            return base_name

        name = base_name
        i = 1
        while name in self.node_labels :
            name = base_name + "_%d" % i 
            i += 1

        return name

    def crawl(
        self,
        roots,
        get_next_fct,
        get_uid_fct,
        get_node_label_fct,
        get_node_parameters_fct=None,
        get_edge_parameters_fct=None,
        parents_to_children=True,
        get_node_attributes_fct = None,
        get_edges_attributes_fct = None,
        autoincrement_names=True,
        reset=False
    ) :

        def _derive(root, nodes, edges, node_labels) :
            root_name = self.resolve_node_name(get_node_label_fct(root), autoincrement_names)
            node_labels.add(root_name)
            root_uid = get_uid_fct(root)
            if root_uid :
                nodes[root_uid] = {
                    "label": root_name,
                    "attributes": {
                        "label": root_name
                    }
                }

                if get_node_attributes_fct :
                    nodes[root_uid]["attributes"].update(get_node_attributes_fct(root))
                self.nodes[root_uid]["label"] = self.nodes[root_uid]["label"] + "(%s)" % ( len(self.nodes[root_uid]["attributes"]) -1 )

                if get_node_parameters_fct :
                    nodes[root_uid].update(get_node_parameters_fct(root))

                for d in get_next_fct(root) :
                    if d is not root :
                        child_uid = get_uid_fct(d)
                        if child_uid :
                            if parents_to_children :
                                con = (root_uid, child_uid)
                            else :
                                con = (child_uid, root_uid)
                            
                            self.edges[con] = {}
                            if get_edges_attributes_fct :
                                self.edges[con]["attributes"] = get_edges_attributes_fct(con[0], con[1])
                            else :
                                self.edges[con]["attributes"] = {}
                            
                            if get_edge_parameters_fct :
                                self.edges[con].update(get_edge_parameters_fct(con[0], con[1]))

                            _derive(d, nodes, edges, node_labels)

        if reset :
            self._init()
    
        for root in roots :
            _derive(root, self.nodes, self.edges, self.node_labels)

class DagreDAG(Abstract_DAG) :
    """"""

    def __init__(self, notebook, name):
        super(DagreDAG, self).__init__(notebook, name)
        self.graph_attributes = {}

    def set_attributes(self, dct) :
        self.graph_attributes = dct

    def reset_css(self) :
        self.css_rules = {}
        self.css_rules["text"] = (
            "font-weight: 300",
            'font-family: "Helvetica Neue", Helvetica, Arial, sans-serif',
            'font-size: 14px'
        )

        self.css_rules[".node rect"] = (
            "stroke: #999",
            'fill: #fff',
            'stroke-width: 1.5px'
        )

        self.css_rules[".edgePath path"] = (
            "stroke: #333",
            'stroke-width: 1.5px'
        )    

    def get_html(self) :
        def _pseudo_jsonify(dct) :
            attrs = [ ]
            for k, v in dct.items() :
                if type(v) is dict :
                    vv = _pseudo_jsonify(v)
                    attrs.append("'%s': {%s}" % (k, vv))
                else :
                    attrs.append("'%s': '%s'" % (k, v))

            return ','.join(attrs)

        def _set_nodes() :
            res = []
            for node_id, params in self.nodes.items() :
                attrs = _pseudo_jsonify(params)
                res.append( "g.setNode('{node_id}', {{ {attributes} }} );".format(node_id = node_id, attributes = attrs ))
            
            return '\n'.join(res)

        def _set_edges() :
            res = []
            for n1, n2 in self.edges :
                res.append( "g.setEdge('%s', '%s')" % (n1, n2) ) ;
            return '\n'.join(res)

        graph_attributes = "{%s}" % _pseudo_jsonify(self.graph_attributes)
        
        template = """
        <script src="../static/libs/d3/js/d3.v4.min.js" charset="utf-8"></script>
        <script src="../static/libs/dagre-d3/js/dagre-d3.js"></script>
        
        <div class="uk-grid">
            <svg class="uk-width-expand@s" id="svg-canvas" ></svg>
            <div class="uk-width-1-2@s uk-width-1-4@m uk-container">
                <div class="uk-card uk-card-default">
                    <h3 class="uk-card-title uk-margin uk-text-center"> Graph attributes</h3>
                    <div class="uk-card-body" id="graph-attributes"></div>
                </div>
                <div class="uk-card uk-card-default">
                    <h3 class="uk-card-title uk-margin uk-text-center"> Node attributes</h3>
                    <div class="uk-card-body" id="node-attributes"></div>
                </div>
            </div>
        </div>


        <script id="js">
            var show_attributes = function(attributes, div_id){{
                html = `<ul class="uk-list uk-list-striped">`
                for (const [key, value] of Object.entries( attributes) ) {{
                    html = html + `<li> ${{key}}: ${{value}}</li>`
                }}
                html = html + '</ul>'
                $('#'+div_id).html(html)
            }}

            show_attributes({graph_attributes}, "graph-attributes")

            // Create the input graph
            var g = new dagreD3.graphlib.Graph()
              .setGraph({{}})
              .setDefaultEdgeLabel(function() {{ return {{}}; }});

            {nodes}
            g.nodes().forEach(function(v) {{
              var node = g.node(v);
              // Round the corners of the nodes
              node.rx = node.ry = 5;
            }});

            {edges}
            // Create the renderer
            var render = new dagreD3.render();

            // Set up an SVG group so that we can translate the final graph.
            var svg = d3.select("svg"),
                svgGroup = svg.append("g");

            // Run the renderer. This is what draws the final graph.
            render(d3.select("svg g"), g);

            svgGroup.selectAll("g.node").on('click', function(name){{ show_attributes(g.node(name)['attributes'], 'node-attributes') }} )

            // Center the graph
            svg.attr("height", g.graph().height + 40);
        </script>""".format(nodes = _set_nodes(), edges= _set_edges(), graph_attributes=graph_attributes)

        return template

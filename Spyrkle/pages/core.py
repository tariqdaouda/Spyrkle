import mistune

from collections import OrderedDict
from . import utils

from ..notebook import Abstract_Section

class Notes(Abstract_Section):
    '''
    Class for notes in the notebook
    notebook: notebook object
    name: string, **kwargs of the section thatr will contain the notes
    notes_html: list, html for corresponding notes output 
    '''
    def __init__(self, **kwargs):
        super(Notes, self).__init__( **kwargs) # the usage of BOOK object is shown in Abstract_Section
        self.notes_html = []

    def add_note_html(self, html) :
        '''Add html for the notes to the html list, register any necessary folders'''
        self.notes_html.append(html)
        
    def add_note_markdown(self, mkd) :
        '''Add markdown for the notes to the html list, register any necessary folders'''
        self.add_note_html(mistune.markdown(mkd))
        
    def add_note(self, title, body, img_src=None, code=None, add_line_reference=True) :
        '''
        Add a note
        title: string, title of the note
        body: string, body of the note
        img_src: filepath for an image that is associated with a note
        code: Any code to be included in a note
        add_line_reference: if True, adds reference to the line of code where note was created in script
        '''
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
                code = "<pre class='uk-text-left'>%s</pre>" % '\n'.join(strip_lines)
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
        '''
        Adds bullet-point notes to a section
        title: string, title of the note
        points: list of strings, each string is list will get a bullet point in a note
        img_src: filepath for an image that is associated with a note
        add_line_reference: if True, adds reference to the line of code where note was created in script
        '''
        
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
        <div class="uk-grid uk-child-width-1-3@m uk-child-width-1-1@s uk-text-center" uk-grid >
        {notes}
        </div>
        """.format(notes = "\n".join(self.notes_html))
        return html

class Articles(Abstract_Section):
    '''
    Used to create articles or long-form messages in a notebook
    article_html: list, to contain associated article html
    '''
    def __init__(self, **kwargs):
        super(Articles, self).__init__( **kwargs)
        self.article_html = [] # article_html contains a list of html

    def add_article(self, title, abstract, body, image) :
        '''
        Adds article to a notebook
        title: string, title of the article
        abstract: string, abstract of an article
        body: string, body of an article
        image: any associated image with an article
        '''
        im = Figure(image)
        html = """
            <h1 class="uk-article-title"><a class="uk-link-reset" href="">{title}</a></h1>
            <p class="uk-text-lead">{abstract}</p>
            <p> {body} </p>
        """.format(title = title, abstract = abstract, body = body)#, src = im.get_src(self.notebook.static_folder))
        self.article_html.append(html) # each add_ functions append a new html into the _html list.

### Making section from txt file ###
    def add_from_doc(self, txtfile) :
        txt = utils.Text(txtfile)
        txt.set_content()
        html = """
            <h1 class="uk-article-title"><a class="uk-link-reset" href="">{title}</a></h1>
            <p class="uk-text-lead">{abstract}</p>
            <div> {body} </div>
        """.format(title = txt.get_title(), abstract = txt.get_abstract(), body = txt.get_body())
        self.article_html.append(html)

    def get_html(self) :
        '''Returns uikit-formatted article html'''
        html="""
        <article class="uk-article">
        {notes}
        </article>
        """.format(notes = "\n".join(self.article_html))
        return html

# class Figures(Abstract_Section):
#     '''In development-support for figures in notebook'''
#     def __init__(self, **kwargs, save_folder = "temp_figs", final_folder = "figs"):
#         import os
#         super(Figures, self).__init__( **kwargs)
#         #self.figure_path = os.path.join(".", notebook.name.replace(" ", "_").lower(), "figs")
#         #self.html_path = os.path.join(".", "figs")
#         self.figure_path = "/".join([".", notebook.name.replace(" ", "_").lower(), save_folder])
#         self.html_path = "/".join([".", final_folder])

#         self.files = []
#         # r=root, d=directories, f = files
#         for r, d, f in os.walk(self.figure_path):
#             for file in f:
#                 if '.png' in file:
#                     self.files.append(file)
#         self.files = [os.path.join(self.html_path, s) for s in self.files]

#     def get_html(self) :
#         fig_html = []
#         for i in self.files:
#             html="""
#             <img src="{img}" alt="" uk-img>
#             """.format(img = i)
#             fig_html.append(html)
#         html_final = """
#         {img}
#         """.format(img = "\n".join(fig_html))
#         return(html_final)

class Figure(Abstract_Section):
    '''Add a figure'''
    def __init__(self, url_or_plot, **kwargs):
        super(Figure, self).__init__( **kwargs)
        self.image = utils.Figure(url_or_plot)

    def get_html(self) :
        src = self.page.notebook.remove_self_url_root(self.image.get_src(self.page.folder))
        if not self.image.is_interactive():
            html="""
                <img src="{image}" alt="{name}" uk-img>
            """.format(image = src, name=self.image.name)
        else :
            html="""
                <iframe src="{image}" style="height:75%;width:100%;"></iframe>
            """.format(image = src)
        return html



class Heading(Abstract_Section):
    '''Add a heading'''
    def __init__(self, text, position="center", line=True, divider=False, bullet=False, size="large", **kwargs):
        super(Heading, self).__init__( **kwargs)
        self.text = text
        self.position = position
        self.line = line
        self.divider = divider
        self.bullet = bullet
        self.size = size

    def get_html(self) :
        classes=["uk-margin-large", "uk-text-%s" % self.position, "uk-heading-%s" % self.size]
        if self.line:
            classes.append("uk-heading-line")

        if self.divider:
            classes.append("uk-heading-divider")

        if self.bullet:
            classes.append("uk-heading-bullet")

        html="""<h1 class="%s"> <span>%s</span> </h1>""" % (" ".join(classes), self.text)
        return html

class HTML(Abstract_Section):
    '''Add an HTML section'''
    def __init__(self, html, **kwargs):
        super(HTML, self).__init__( **kwargs)
        self.html = html

    def get_html(self) :
        return self.html

class PandasDF(HTML):
    '''Add pandas Dataframe'''
    def __init__(self, df, **kwargs):
        super(PandasDF, self).__init__( html=df.to_html(), **kwargs)

class Code(HTML):
    '''Add some code'''
    def __init__(self, code, **kwargs):
        html = "<pre class='uk-text-left'>%s</pre>" % code
        super(Code, self).__init__( html=html, **kwargs)

class Table(Abstract_Section):
    def __init__(self, data, header=None, **kwargs):
        """Add a tabe. Expects a list of dict for data. Header is provided should be a list of strings"""
        super(Table, self).__init__( **kwargs)
        self.header = header
        self.data = data

    def get_html(self) :
        '''Returns html'''
        
        def _make(header, data):
            body = []
            for line in data :
                res = []
                for k in header :
                    try:
                        res.append( str(line[k]) )                    
                    except KeyError :
                        res.append("NA")
                body.append(res)

            header = "<th>" + "</th>\n<th>".join(header) + "</th>"
            body = "<tr>" + "</tr>\n<tr>".join(
                    [ 
                        "<td>" + "</td>\n<td>".join(line) + "</td>" for line in body
                    ]
                ) + "</tr>\n"
            return header, body

        if self.header is not None :
            header = self.header
        else :
            header = OrderedDict()
            for line in self.data:
                for key in line.keys() :
                    header[key] = True
            
            header = list(header.keys())

        header, body = _make(header, self.data)
       
        html = """
        <table class="uk-table uk-table-striped">
            <thead>
                {header}
            </thead>
            <tbody>
                {body}
            </tbody>
        </table>
        """.format(header = header, body = body)

        return html

class Caption(Abstract_Section):
    def __init__(self, mkd_title, mkd_caption, **kwargs):
        super(Caption, self).__init__( **kwargs)
        self.title = mistune.markdown(mkd_title)
        self.caption = mistune.markdown(mkd_caption)

    def get_html(self) :
        '''Returns markdown html'''
        html = """
            <article class="uk-article-title">
                <p class="uk-text-lead">{title}</p>
                <p class="uk-text-meta">{caption}</p>
            </article>
        """.format( title = self.title[3:-5], caption=self.caption[3:-5])
        # print(html)
        return html

class Markdown(Abstract_Section):
    def __init__(self, text, **kwargs):
        super(Markdown, self).__init__( **kwargs)
        self.text = text

    def get_html(self) :
        '''Returns markdown html'''
        return mistune.markdown(self.text)


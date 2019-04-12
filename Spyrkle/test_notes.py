import notebook

book = notebook.Notebook("test_notebook")

notes = notebook.Notes(book, "Notes on life")
for i in range(10) :
	notes.add_note("Note %s" % i, "Life is %s Lorem ipsum dolor sit amet, consectetur adipisicing elit. Alias dolorum asperiores at veritatis architecto sequi nulla perspiciatis rerum modi, repellat assumenda quisquam dolorem sit molestiae aspernatur cum nemo placeat laboriosam." % i)

notes = notebook.Notes(book, "Notes on life 2")
for i in range(10) :
	notes.add_note("Note %s" % (i+10), "Life is %s Lorem ipsum dolor sit amet, consectetur adipisicing elit. Alias dolorum asperiores at veritatis architecto sequi nulla perspiciatis rerum modi, repellat assumenda quisquam dolorem sit molestiae aspernatur cum nemo placeat laboriosam." % i)

notes = notebook.Notes(book, "Notes on life 3")
for i in range(10) :
	notes.add_bullet_points_note("Note %s" % (i+100), ["test", "text", "iop"])

book.save()
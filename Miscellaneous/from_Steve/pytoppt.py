import os

# You have to use makepy.py in site-packages/win32com/client/makepy.py to create
# the modules MSO and MSPPT.  Use the -d flag when you run it to choose from a list.
# Run it twice.  Choose Microsoft Objects the first time and then Microsoft Powerpoint 
# Objects the next.  I had to rename the directories that makepy.py creates from some
# stupid long name to MSO and MSPPT.

# For Microsoft Office 2010/2012 you may need to cd into the objects folder and run the make.py script 
# on the particular file you are interested in.

import win32com.client  # middleman/translator/messenger between windows and python
import win32com.gen_py.MSO as MSO  # contains constants referring to Microsoft Office Objects
import win32com.gen_py.MSPPT as MSPPT  # contains constants referring to Microsoft Office Power Point Objects

g = globals()  # a dictionary of global values, that will be the constants of the two previous imports

for c in dir(MSO.constants):
    g[c] = getattr(MSO.constants, c)  # globally define these

for c in dir(MSPPT.constants):
    g[c] = getattr(MSPPT.constants, c)

Application = win32com.client.Dispatch("PowerPoint.Application")
Application.Visible = True  # shows what's happening, not required, but helpful for now

Presentation = Application.Presentations.Add()  # adds a new presentation

#fullTemplatePath = "TSC_Template.potx"
fullTemplatePath = "C:\\AnalysisTools\\pythonTools\\Miscellaneous\\from_Steve\\TSC_Template.potx"

titleForTitleSlide = "(U) This is a title"
subtitleForTitleSlide = "22 October 2013" + os.linesep + "Stephen McMurray"

#ImageToSlides = [	{"Title": "(U) Range v. Time", "Image": "biconic_130.gif"},
#                 {"Title": "(U) Range v. Time" + os.linesep + "Tracks of Interest", "Image": "biconic_130.gif"}
#                ]
ImageToSlides = [{"Title": "(U) Range v. Time", "Image": "C:\\AnalysisTools\\pythonTools\\Miscellaneous\\from_Steve\\biconic_130.gif"},
                 {"Title": "(U) Range v. Time" + os.linesep + "Tracks of Interest", "Image": "C:\\AnalysisTools\\pythonTools\\Miscellaneous\\from_Steve\\biconic_130.gif"}
                ]

slideNum = 1
TitleSlide = Presentation.Slides.Add(slideNum, ppLayoutTitle)
TitleSlide.Shapes(1).TextFrame.TextRange = titleForTitleSlide
TitleSlide.Shapes(2).TextFrame.TextRange = subtitleForTitleSlide

# Slide1 = Presentation.Slides.Add(1, ppLayoutBlank) # new slide, at beginning
# Slide1.Shapes.AddShape(msoShapeRectangle, 100, 100, 200, 200)
for slide in ImageToSlides:
    slideNum += 1
    title = slide["Title"]
    pictName = slide["Image"]

    NewSlide = Presentation.Slides.Add(slideNum, ppLayoutTitleOnly)
    NewSlide.Shapes(1).TextFrame.TextRange = title
    #Incidentally, the dimensions are in points (1/72").  Since the default
    #presentation is 10" x 7.5" the size of each page is 720x540.  The plots are
    #20"x11" so that would be (20.*72.) x (11.*72.)
    #Then we want to shrink by 50% or (10.*72.) x (5.5*72.)
    #Horizontal is 0, Vertical is -1.67" so 1.67*72

    left = 0.75*72
    top = 2.08*72
    width = 8.42*72.
    height = 4.63*72
    Pict1 = NewSlide.Shapes.AddPicture(FileName=pictName, LinkToFile=False, SaveWithDocument=True, Left=left, Top=top, Width=width, Height=height)

    rectH = 5.17 * 72
    rectW = 9. * 72
    rectTop = 1.83 * 72
    rectLeft = 0.5 * 72
    Rect1 = NewSlide.Shapes.AddShape(Type=msoShapeRectangle, Left=rectLeft, Top=rectTop, Width=rectW, Height=rectH)
    Rect1.Fill.Visible = msoFalse
    #Rect1.Line.ForeColor.RGB = rgbRed
    # Rect1.Line.ForeColor.RGB = rgbGreen
    textH = 0.24 * 72
    textW = 1.1 * 72
    textTop = 1.83 * 72
    textLeft = 0.5 * 72
    Text1 = NewSlide.Shapes.AddTextBox(msoTextOrientationHorizontal, textLeft, textTop, textW, textH)
    Text1.TextFrame.TextRange.Text = "UNCLASSIFIED"
    Text1.TextFrame.TextRange.Font.Size = 8
    Text1.TextFrame.TextRange.Font.Bold = msoTrue
    #Text1.TextFrame.TextRange.Font.Color.RGB = rgbRed
    # Text1.TextFrame.TextRange.Font.Color.RGB = rgbGreen
    textH = 0.24 * 72
    textW = 1.1 * 72
    textTop = 6.75 * 72
    textLeft = 8.38 * 72
    Text2 = NewSlide.Shapes.AddTextBox(msoTextOrientationHorizontal, textLeft, textTop, textW, textH)
    Text2.TextFrame.TextRange.Text = "UNCLASSIFIED"
    Text2.TextFrame.TextRange.Font.Size = 8
    Text2.TextFrame.TextRange.Font.Bold = msoTrue
    #Text2.TextFrame.TextRange.Font.Color.RGB = rgbRed
    # Text2.TextFrame.TextRange.Font.Color.RGB = rgbGreen

Presentation.ApplyTemplate(fullTemplatePath)
#Presentation.SlideMaster.ApplyTheme(fullTemplatePath)

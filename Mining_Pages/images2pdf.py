import img2pdf

import os


#pdf = FPDF(unit="pt")
# imagelist is the list with all image filenames
inputdir='X:/Projekte/iDAI.shapes/Mining_Shapes/INCOMING/testfpdf'
for folder in os.listdir(inputdir):

    imagelist = os.listdir(os.path.join(inputdir,folder))
    print(imagelist)
    imagelistpath =[]
    for image in imagelist:
        if image.endswith(".jpg"):
            imagelistpath.append(os.path.join(inputdir,folder,image))

    with open(inputdir + "/name.pdf","wb") as f:
	    f.write(img2pdf.convert(imagelistpath))
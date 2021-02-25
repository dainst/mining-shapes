import img2pdf
#from pdf2image import convert_from_path
import os


def findstem(arr):

    # Determine size of the array
    n = len(arr)

    # Take first word from array
    # as reference
    s = arr[0]
    l = len(s)

    res = ""

    for i in range(l):
        for j in range(i + 1, l + 1):

            # generating all possible substrings
            # of our reference string arr[0] i.e s
            stem = s[i:j]
            k = 1
            for k in range(1, n):

                # Check if the generated stem is
                # common to all words
                if stem not in arr[k]:
                    break

            # If current substring is present in
            # all strings and its length is greater
            # than current result
            if (k + 1 == n and len(res) < len(stem)):
                res = stem

    return res


#pdf = FPDF(unit="pt")
# imagelist is the list with all image filenames






inputdir = 'U:/Projekte/iDAI.shapes/Mining_Shapes/INCOMING/Img2pdf'
print(img2pdf --help)
#img2pdf.convert'U:/Projekte/iDAI.shapes/Mining_Shapes/INCOMING/Img2pdfZenonID_000587058/BaM17_.pdf' tmp
#convert_from_path(os.path.join(inputdir,'ZenonID_000587058/BaM17_.pdf'), dpi=300, fmt='png', thread_count=4, output_file=os.path.join(inputdir,'ZenonID_000587058/BaM17_.pdf'), output_folder=os.path.join(inputdir,'/ZenonID_000587058/'))
#for folder in os.listdir(inputdir):

    #imagelist = os.listdir(os.path.join(inputdir, folder))
    #print(imagelist)
    #imagelistpath = []
    #imageliststem = []
    #for image in imagelist:
        #if image.endswith(".jpg"):
            #imagelistpath.append(os.path.join(inputdir, folder, image))
            #imageliststem.append(image)
    #stem = findstem(imageliststem)
    #print(imageliststem)
    #print(stem)
    #stem = stem.replace('.jpg', '')
    #with open(inputdir + "/" + folder + '/' + stem + ".pdf", "wb") as f:
        #f.write(img2pdf.convert(imagelistpath))

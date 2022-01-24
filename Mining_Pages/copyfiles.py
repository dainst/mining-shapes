from shutil import copyfile
import os
sourcefolder="C:/Users/mhaibt/AppData/Roaming/idai-field-client/imagestore/idaishapes_test/"
targetfolder="C:/Users/mhaibt/Documents/GitHub/idai-field/web/data/cantaloupe/"

for file in os.listdir(sourcefolder):
    
    filename=os.path.basename(file)
    print('Source: ', sourcefolder + file, ' Target: ', targetfolder + filename + '.png' )

    copyfile( sourcefolder + file, targetfolder + filename  )
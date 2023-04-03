import zipfile
zip_ref = zipfile.ZipFile("CXR-Classification-using-mobilenetv3small/model_mobilenet.zip")
zip_ref.extractall()
zip_ref.close()

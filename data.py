from roboflow import Roboflow
rf = Roboflow(api_key="RDw5cvxZyuUsCtn4nBch")
project = rf.workspace("ishwar-g3pdo").project("dataset-aucqa")
version = project.version(9)
dataset = version.download("yolov9")





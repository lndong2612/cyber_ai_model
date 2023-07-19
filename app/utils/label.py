from app.database.label_db import Label


def convert_id2label(id):
    label_db = Label.get(id)
    if label_db:
        return label_db.labelName
    else:
        return 'unknown'
    
def convet_list_label_id(label_id_list):
    res = []
    for label_id in label_id_list:
        label_name = convert_id2label(label_id)
        res.append(label_name)
    return res

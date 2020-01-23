import json

with open('TeacherAI/tai-documents-v3.json') as f:
    data = json.load(f)

good_leads = [i for i in data if i["grades"][1]["score"]["criteria"]["lead"] == 1.0]

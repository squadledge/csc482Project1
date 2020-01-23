import json

with open('TeacherAI/tai-documents-v3.json') as f:
    data = json.load(f)

good_leads = [i for i in data if i["grades"][1]["score"]["criteria"]["lead"] >= 4.0]
print(good_leads)

with open('good_leads.txt','w') as f:
    for i in good_leads:
        json.dump(i,f)
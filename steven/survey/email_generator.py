import csv

# takes the distribution csv from qualitrics and generates emails 

# initial sheet with group and SA assignments, with emails in name col
group_sheet_name = 'old_groups_test.csv'
# distribution sheets for part 1 and 2 downloaded from Qualitrics
dist1_name = 'Strategic_Behavior_Activity_Part_1-Distribution_History_test.csv'
dist2_name = 'Strategic_Behavior_Activity_Part_2-Distribution_History_test.csv'
# template for emails
email_template_name = 'email_template.html'

with open(group_sheet_name, newline='') as infile:
    r = csv.reader(infile)
    header = next(r)
    assert(all([x == y for x, y in 
        zip(header, ['name', 'sas', 'authored_sa', 'authored_id', 'target_sa', 'target_id', 'group'])
        ]))
    group_to_names = {}
    name_to_group = {}
    for row in r:
        name = row[0].strip()
        group = int(row[6])
        if group in group_to_names:
            group_to_names[group].append(name)
        else:
            group_to_names[group] = [name]
        name_to_group[name] = group

with open(dist1_name, newline='') as infile1, open(dist2_name, newline='') as infile2, open(email_template_name, newline='') as email_template_file:
    r1 = csv.reader(infile1)
    r2 = csv.reader(infile2)
    email_template = email_template_file.read()
    header = next(r1)
    header2 = next(r2)
    assert(header[4] == 'Email' and header[7] == 'Link')
    assert(header2[4] == 'Email' and header2[7] == 'Link')
    for row1, row2 in zip(r1, r2):
        assert(row1[4] == row2[4])
        email = row1[4]
        link1 = row1[7]
        link2 = row2[7]

        group = name_to_group[email]
        group_members = group_to_names[group].copy()
        group_members.remove(email)

        if len(group_members) == 0:
            parts = email_template.split('$')
            email_text = parts[0] + parts[2]
            email_final = email_text.format(link1=link1, link2=link2)
        else:
            email_text= email_template.replace('$', '')
            group_email_list = [n for n in group_members]
            group_emails = ', '.join(group_email_list)
            email_final = email_text.format(link1=link1, link2=link2, group_emails=group_emails)

        name = email.split('@')[0].strip()
        with open(name + ".html", 'w') as outfile:
            outfile.write(email_final)

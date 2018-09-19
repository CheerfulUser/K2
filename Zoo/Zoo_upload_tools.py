def Save_space(Save):
    try:
        if not os.path.exists(Save):
            os.makedirs(Save)
    except FileExistsError:
        pass

def List_gaps(List, Space):
    '''
    Makes spaces of specified size in a list.
    '''
    for i in range(Space):
        List.append('')
    return List

def Zoo_ticket(Set_name, Dictionary, Path, Permit, Prohibit, Save):
    '''
    Makes a ticket that contains the following information from a Zoo upload:
    Set name
    Campaign
    Permited types
    Prohibited types
    Files uploaded
    '''
    from datetime import datetime
    ticket = []
    
    ticket.append(Set_name)
    ticket = List_gaps(ticket,1)
    
    ticket.append('Campaign')
    camp = 'C' + Path.split('c')[-1].split('/')[0]
    ticket.append(camp)
    ticket = List_gaps(ticket,1)
    
    ticket.append('Permited')
    for i in range(len(Permit)):
        ticket.append(Permit[i])
    ticket = List_gaps(ticket,2)
    
    ticket.append('Prohibited')
    for i in range(len(Prohibit)):
        ticket.append(Prohibit[i])
    ticket = List_gaps(ticket,2)
    
    ticket.append('Files')
    for filename, metadata in Dictionary.items():
        ticket.append(filename)
    
    Save_space(Save)
    
    savename = (Save + 'Zoo_upload_' + str(datetime.now().year) + str(datetime.now().month) 
                + str(datetime.now().day) + str(datetime.now().hour) + str(datetime.now().minute) 
                + str(datetime.now().seconds) + '.txt')
    
    with open(savename, 'w') as f:
        for item in ticket:
            f.write("%s\n" % item)
    

def Zoo_upload(Set_name, Path, Permit, Prohibit, Save):
    '''
    Uploads files to the Zooniverse server for the K2:BS project. 
    Set_name is the name of the uploaded set (can't be the same as a previous set). 
    Path is the path to the parent directory containing the Events.csv file.
    Permit is the permited event types.
    Prohibit is the prohibited event types.
    Save is the location where the Zoo upload ticket will be saved.
    '''
    from panoptes_client import Panoptes, Project, SubjectSet, Subject
    import pandas as pd
    import os
    # Connect to my account
    Panoptes.connect(username='cheerfuluser',password='dragon1243')
    # Link the K2BS project
    
    project = Project.find(slug='cheerfuluser/k2-background-survey')
    # Make the new subject set
    subject_set = SubjectSet()
    subject_set.links.project = project
    subject_set.display_name = Set_name
    subject_set.save()

    # Load event paths and metadata
    events = pd.read_csv(Path + 'Events.csv',header=0)
    ar = events.values # Array form of events
    keys = events.keys() # Header entries 
    # Organise files and metadata into the Zoo example
    dic = {}
    failed = {}
    for i in range(len(events.Counts)):
        if os.path.isfile(Path + ar[i,14][2:]):
            # Conduct a check to only let through preselected types specified in the Uploads variable
            allowed = False
            for j in range(len(Permit)):
                if (Permit[j] in (Path + ar[i,14][2:])):
                    allowed = True
                    
            for j in range(len(Prohibit)):
                if (Prohibit[j] in (Path + ar[i,14][2:])):
                    allowed = False
            
            if allowed == True:
                entry = {}
                entry = {Path + ar[i,14][2:] : {
                    keys[0] : ar[i,0],
                    keys[1] : ar[i,1],
                    keys[2] : ar[i,2],
                    keys[3] : ar[i,3],
                    keys[4] : ar[i,4],
                    keys[5] : ar[i,5],
                    keys[6] : ar[i,6],
                    keys[7] : ar[i,7],
                    keys[8] : ar[i,8],
                    keys[9] : ar[i,9],
                    keys[10] : ar[i,10],
                    keys[11] : ar[i,11],
                    keys[12] : ar[i,12],
                    keys[13] : ar[i,13]}
                        }
                dic.update(entry)
        else:
            entry = {}
            entry = {Path + ar[i,14][2:] : {
                keys[0] : ar[i,0],
                keys[1] : ar[i,1],
                keys[2] : ar[i,2],
                keys[3] : ar[i,3],
                keys[4] : ar[i,4],
                keys[5] : ar[i,5],
                keys[6] : ar[i,6],
                keys[7] : ar[i,7],
                keys[8] : ar[i,8],
                keys[9] : ar[i,9],
                keys[10] : ar[i,10],
                keys[11] : ar[i,11],
                keys[12] : ar[i,12],
                keys[13] : ar[i,13]}
                    }
            failed.update(entry)

    new_subjects = []
    
    for filename, metadata in dic.items():
        print(filename)
        subject = Subject()

        subject.links.project = project
        subject.add_location(filename)

        subject.metadata.update(metadata)

        subject.save()
        new_subjects.append(subject)
    
    subject_set.add(new_subjects)
    workflow = project.links.workflows[0]
    workflow.add_subject_sets(subject_set)
    
    Zoo_ticket(Set_name,dic,Path,Permit,Prohibit,Save)
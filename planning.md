# Project management

We will use Github Issues, and the board in the Projects tab, to manage the project. Any coding-related to-do's (for example, implementing a function) should be created as an issue, because this allows adding comments, labels, deadlines etc. Then from the project board, you can import this issue as a card. 

A good to-do should be actionable (start with a verb, such as implement or read) and concrete (for example, a specific functionality, a number of papers, etc). 

Cards start in the "backlog" column, and then progress through "to-do" (plan for next week), "in progress" (doing it now) and "done". The idea is to have as few things as possible in the "in progress" column, and to prioritize what you have in "to-do". This is based on the agile or Kanban methdology. You might want to search for "examples Kanban research projects" or similar queries to get an idea of how people use it for such projects. 

## Data storage

We can store datasets and other large files on the Open Science Framework website, in this project: https://osf.io/x2fpg/ . It is in some ways similar to Github, because there is version control, contributors, etc. The storage is unlimited, but each file is limited to 5 GB, and you can only upload files, not folders. Therefore it will be suitable for trained models, or ZIP files with images (provided these are less than 5GB). 

A useful (free) course on data management: https://www.coursera.org/learn/data-management  Please look at a few videos regarding for example file naming. 

An example of organizing files: https://medium.com/@rrfd/cookiecutter-data-science-organize-your-projects-atom-and-jupyter-2be7862f487e

## Overall plan

* Weeks 1-2 - Reproduce a result from previous work (already pretrained network + existing data), setup Sacred 

* Weeks 3-4 (project 1) - Extend train/test setup, setup code so adding datasets is easy
* Weeks 3-4 (project 2) - Start implementing similarity measures (possibly also ways to compare similarity matrices)

* Weeks 5-6 (project 1) - Extend to varying dataset characteristics, store results in Sacred  
* Weeks 5-6 (project 2) - Extract results/ground truth from Sacred, compare to implemented similarity measures

* Weeks 7-8 - Finish experiments / documentation / etc. 

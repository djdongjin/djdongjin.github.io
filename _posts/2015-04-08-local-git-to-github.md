---
layout: post
title: "Upload a local git repo to GitHub"
categories: []
---


1. create a git repository in your GitHub
2. create a local git repository whose name is the same with the git created in 1, and initialized by 'git init'
3. **'git pull *github repository url*'** to use the readme.md/.gitignore files created by GitHub
4. **'git add .'** to add files needed to be updated
5. **'git commit -m 'commit'** to commit an update
6. **'git remote add origin *github repo url*'** to link the remote repo with the local repo
7. **'git push -u origin master'** to push our update

After these step, next time we need to push an update, we just need to do as step 4, 5, 7, that's, add-commit-push



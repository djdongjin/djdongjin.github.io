---
layout: post
title: "Note for Git"
categories: [Basics]
---

**集中式版本控制系统**: 版本库存放在中央server, 工作时首先*联网*从中央server取得最新版本, 任务完成后再把改动推送回中央server. 最大的弊端就在于必须联网才能工作.

**分布式版本控制系统**: 无中央server, 每台电脑均保存完整的版本库, 因此工作时*不需要联网*, 多人协作时只需要把各自改动推送给对方. 由于不同电脑不在同一LAN, 推送改动不方便, 因此一般也有一台电脑充当”中央server”, 不过其作用仅限于方便交换大家的修改.

安装好git后首先设置name和email:
```bash
git config --global user.name "name"
git config --global user.email "xx@xx.com"
```

## 创建版本库, 提交修改

**创建Git版本库**: 版本库也叫作仓库(repository). 在项目根目录执行:
```bash
git init
```

> 版本控制系统只能跟踪文本文件(如txt, .java)的改动; 对于图片(包括word)等二进制文件, 只能把每次改动后的文件串起来, 而无法知道具体改了什么.

**添加文件到版本库**: 第一步将文件添加到仓库; 第二步告诉git把文件提交到仓库:
```bash
git add readme.txt  # 可多次调用, 然后通过commit一起提交
git commit -m "write a comment here"
```

**查看仓库状态**: git status可告诉我们当前仓库的状态, 如修改的文件, 待提交的文件
```
git status
```

**查看修改前后异同**: 可查看修改待提交的文件具体修改的内容:
```
git diff readme.txt
```

**查看历史记录**: git log记录每次提交的**版本号**(commit id), author, date, comment:
```
git log [--pretty=oneline (简化output)]
```

> Git相当于把不同版本串成一条时间线.

## 版本管理

**版本回退**: 回退使用git reset命令. *HEAD*表示当前版本, *HEAD^*表示上一个版本, 有多少^就表示上多少个版本, 当^过多时, 如100, 可写成*HEAD\~100*:
```
git reset --hard HEAD^
# 如果想再回到最新的版本, 需要记得对应版本的commit id(不用记完整的)
git reset --hard 1094a
```

> Git内部有指向当前版本的HEAD指针, 进行回退时, 只是修改HEAD指针, 因此回退速度很快.

如果实在记不得commit id, 可通过git reflog查询每一次命令, 里边有相应commit id:
```
git reflog
git reset --hard xxx
```

**工作区(Working Directory)**: 即电脑上Git所管理项目所在的文件夹.
**版本库(Repository)**: 工作区内的*.git*文件夹, 即为Git的版本库, 其内容最重要的就是称为stage(index)的*暂存区*, 还有git自动创建的第一个分支*master*以及指向master的指针*HEAD*.

在将文件添加到Git版本库时, 第一步git add便是把文件修改添加到暂存区; 第二步git commit便是把暂存区的所有内容提交到当前分支.

**管理修改**: Git追踪并管理的是修改(即我们对文件进行的一次次的操作, 如删除一行)而非文件. 即, 每次通过commit提交的改动, 是以修改为单位, 而非以文件为单位. 比如说, 先在一个文件中添加一行, 然后add, 再在该文件中添加一行后commit. 由于第二次修改未add到暂存区stage中, 故此次commit只会提交第一次修改, 而第二次修改仍处于待添加到stage的状态.

**撤销修改**: git checkout -- \<file\>可令修改后未add的文件回到和版本库一样的状态, 令已add到stage后又做修改的文件回到和add到stage后的状态. 即让文件恢复到最后一次git commit或者git add时的状态.
```
git checkout -- readme.txt
```

针对已经add到stage但是还未commit的修改, 通过如下方式修改:
```
git reset HEAD <file>
```
这样可以把stage的修改撤销掉(unstage), 重新放回工作区. 此时stage是干净的, 但是工作区有修改, 因为上述命令只相当于把stage内的修改打回工作区. 之后就可以通过上个命令真正的撤销修改:
```
git checkout -- readme.txt
```

如果修改已经commit到了版本库, 且还未推送到远程, 可通过上一节版本回退的方法回退到上一版本.

**删除文件**: 删除也是一个修改操作, 如果在工作区删除了一个文件(如rm \<file\>), 可通过如下方式将其也在版本库中删除:
```
git rm <file>
git commit -m "delete a file"
```

如果是误删了文件, 想将其恢复到工作区中, 可通过如下方式:
```
git checkout -- <file>
```

可以发现, git checkout命令就是用版本库的文件来代替工作区的版本, 无论工作区是删除还是修改, 均可将其恢复到原始状态.

## 远程仓库

分布式版本控制的实际工作流程: 找一台电脑充当server(如Github提供的Git仓库托管服务), 其他人都从这个server仓库克隆一份到电脑上工作, 并把各自的推送提交到server仓库里, 同时也从server仓库里再克隆别人的提交. 而server来接受或者拒绝其他人推送的提交, 从而保证server仓库保持更新.

本地Git仓库和GitHub仓库之间通过SSH加密传输, 先进行如下设置:

1. 创建SSH Key:`ssh-keygen -t rsa -C "xxx@xxx.com"`.
2. 在\~/.ssh/目录下有id\_rsa, id\_rsa.pub两个文件, 前者是私钥, 后者是公钥.
3. 登录GitHub, 在Account setting的SSH Keys页面添加SSH Key, Title任意, 内容复制公钥中的内容.

### 添加远程库

已经在本地建立Git仓库, 想在GitHub建立Git仓库, 让两仓库进行远程同步. GitHub仓库既可作为备份, 又可让其他人通过该仓库协作. 操作过程如下:

首先, 在GitHub创建一个新repo, Repository name保持和本地git一直, 其它默认值. 此时GitHub仓库还是空的. 我们可以把它作为一个新仓库开始进行工作, 也可以将其和一个本地仓库关联, 并且把本地仓库的内容推送到该GitHub仓库(我们的目的).

现在需要在本地仓库下运行:
```
git remote add origin git@github.com:<github username>/<repo name>.git
```
添加后, 远程库的名字即为origin(git默认叫法).

接下来需要把本地库的内容推送到远程库:
```
git push -u origin master
```
该命令实际是把当前分支*master*推送到远程. 因为是第一次推送master分支, 因此使用*-u*参数, 在推送本地master分支到远程新的master分支的同时, 还会关联这两个分支, 从而简化以后的推送或者拉取. 比如, 之后再把本地做的提交可通过如下命令把本地master最新修改推送到GitHub:
```
git push origin master
```

### 从远程库克隆

上一节讲的是先有本地库, 后有远程库, 如何关联远程库. 如果从零开始, 最好的方式是先建立远程库, 然后把远程库克隆到本地. (远程库起到server的角色, 而本地库只是一个工作者, 因此这种方式更直观.)

首先按照一样的方式先在GitHub创立远程库. 然后使用git clone克隆一个本地库:
```
git clone git@github.com:djdongjin/gitskills.git
# 地址还可使用https://github.com/djdongjin/gitskills.git
# 前者使用ssh协议, 速度最快; 后者使用https协议, 需要每次输入口令
```

## 分支管理

分支适用于多人协作. 工作时创建自己的分支, 随时提交自己的工作, 而别人仍基于master分支工作. 你的分支目前对他人透明, 直到开发完毕, 一次性合并到原分支, 对他人才可见. 这样, 既方便自己工作, 又不影响别人工作, 提高开发安全.

### 创建与合并分支(Branch)

版本回退章节提到的每次提交串起来的一条时间线, 这条时间线就是一个分支. 目前我们的Git里只有一条时间线/分支, 这个分支也就是主分支master. 每次提交, master分支都会前移一步, 随着提交, master分支线也越来越长. 严格来说, HEAD并不是指向提交, 而是指向master, master才是指向提交的, 因此, HEAD指向的其实是当前分支. 这样的话, 通过HEAD便能确定当前分支, 以及当前分支的提交点.

当创建新的分支(比如dev)时, Git仅仅新建一个指针dev, 指向和master相同的提交, 再让HEAD指向dev, 表示当前分支在dev. 由于仅涉及指针操作, 因此创建分支很快. 此后的提交便都基于dev, 每提交一次, dev和HEAD前移一步, 而master不变. 因此在他人看来版本库还是原来的样子.

如果我们在dev分支完成了相应的工作并且可以发布了, 便可以把dev分支合并到master上. 合并十分简单, 只需要把master指向dev的当前提交. 如果不需要在dev分支继续工作, 便可以删除dev分支了. 删除分支也只需要把dev指针删除.

相关命令以及实际工作流程如下:
**创建并切换到新分支**: 以dev为例:
```
git checkout -b dev
```
该命令等效于如下两条命令:
```
git branch dev    #新建分支
git checkout dev  #切换分支
```
**查看当前所有分支**:
```
git branch
```
该命令列出所有分支, 当前所在分支前会标有”\*”号.

之后我们便可以在dev分支正常修改文件, 并提交到版本库, 推送到远程库. 这些操作均不会影响master分支. 当我们使用`git checkout master`切换回master分支后也会发现, 在dev所提交修改在master分支均未体现, 因为master此时仍处于之前的提交点.

**合并分支**: git merge命令将指定分支合并到当前分支. 我们先切换回master分支之后将dev合并到master分支:
```
git checkout master
git merge dev
```
合并后输出显示该次合并使用的是”Fast-forward”, 即”快进模式”, 也就是直接把master指向dev的当前提交, 合并速度很快. 不过仅适用于无冲突的情况.

**删除分支**: 合并完成后, 边可以删除用于开发的dev分支了:
```
git branch -d dev
```

由于分支的创建合并删除均十分迅速, 因此Git鼓励使用分支完成某个任务, 合并再删除该分支. 这样和直接在master工作效果相同, 但是更安全.

### 解决冲突

假设我们新建并切换到一个工作分支(如feature1), 并且提交一个修改. 之后切换回master分支, 也提交一个修改. 此时两个分支分别处于各自新的提交点. 此时无法执行”Fast-forward”, 只能试图把各自的修改合并, 此时就可能产生冲突(如对一个文件的同一句话进行不同的修改).

当我们试图用`git merge <branch-name>`合并两个分支时, 如果产生冲突, 该命令就会提示自动合并失败, 以及产生冲突的文件(git status也会显示冲突的文件). 查看冲突文件时, 会发现Git使用*\<\<\<\<\<\<\<*, *=======*, *\>\>\>\>\>\>\>*标记处不同分支的内容, 方便我们选择合适的修改版本. 修改后便可以保存并提交了:
```
git add <confict-file>
git commit -m "confict fixed"
```

此时可以用如下命令查看分支合并图:
```
git log --graph
```

可以发现, Git处理冲突的方式为, 如果检测到冲突, Git会把各分支冲突文件中不同的部分总结到该文件中显示, 让用户自己去解决冲突. 解决冲突的过程便可看做是一个正常的修改, 修改完成后便可以正常加入到stage并提交到版本库.

### 分支管理策略

合并分支时, Git会尽量使用”Fast-forward”模式, 但该模式下, 删除分支会丢掉分支信息(即好像该分支从来未存在过, 也不会在git log中显示该分支相关修改信息). 合并时可通过--no-ff选项关闭ff模式, 此时merge时会生成一个新的commit, 也会看出分支信息.
```
git merge --no-ff -m "merge with no-ff" dev
```

**分支策略**: *master*分支应最稳定, 只用来发布新版本, 不能用来干活; 干活应新建一个*dev*分支, 因此*dev*分支变化最频繁, 当某个版本经过测试可发布了, 再将dev的版本merge到master上; 如果多人协作, 每人应再基于dev新建一个自己的工作分支, 时不时往dev分支上合并, 这些都是开发过程中的一些小版本, 因此不需要合并到master分支.

### Bug分支

场景: 有一个bug需要在两小时内修复, 想创建一个临时分支*issue-101*来修复bug, 但是当前dev分支上进行的工作还无法提交(需要1天才可完成).

解决: Git的*stash*功能可以**”储存”工作现场**, 等以后恢复现场后可继续工作:
```
git stash
```
使用该命令后查看工作区状态, 便是干净的了. 可以从master创建临时分支修复bug了. 修复完之后再切换到dev分支继续工作, 此时工作区是干净的, 需要恢复到之前的工作现场.

**查看工作现场列表**:
```
git stash list
```

**恢复工作现场**: 有两种方式:
```
# 1.
git stash apply # 恢复现场
git stash drop  # 删除stash中的内容
# 2.
git stash pop   # 恢复的同时删除
```

### Feature分支

工作时我们主要处于dev分支(如果单独工作), 或者基于dev的个人分支(如果多人协作). 当添加一个功能时, 最好新建一个feature分支, 在上面完成该feature的工作, 再合并到工作分支, 并删除该feature分支.

完成该feature后便可切换到dev准备merge, 进行正常的合并, 删除feature分支. 但另一种情况可能是完成后发现该feature没用(或者经费不足等等)需要取消并删除相关文件, 因此需要直接删除该分支:
```
git branch -d feature-vulcan
```
上述命令会操作失败, 且提示该分支未被合并, 如果删除将丢失修改, 因此需要强制删除, 使用参数-D:
```
git branch -D feature-vulcan
```

### 多人协作

通过`git clone`克隆远程库时, git会自动关联本地的*master*分支和远程的*master*分支, 且远程库的默认名称是*origin*.

**查看远程库信息**:
```
git remote
git remote -v    # 更详细信息
```
会显示可以抓取(fetch)和推送的*origin*地址. 若无push权限, 就看不到push地址.

**推送分支**: 把本地库某分支的所有提交推送到远程库, 需要指定本地分支:
```
git push origin master
git push origin dev
```

> 是否应该推送?
> master是主分支, 因此需时刻和远程同步;
> dev是开发分支, 团队所有成员都在该分支工作, 也需要与远程同步;
> bug只用于本地修复bug然后merge到master, 因此不需推送;
> feature是否推送取决于是否和他人一起完成该feathre.

**抓取分支**: 多人协作时, 大家都往origin的master和dev推送自己的修改. 先假设有两人开发, A(我), B(协作者). A本地库和远程库均有master和dev两个分支. 当B通过`git clone`远程库时, 默认只会克隆master一个分支到本地的master分支. 如果B要在dev上开发, 他需要先*创建远程origin的dev分支到本地*:
```
git checkout -b dev origin/dev
```

B现在可以在本地dev工作并时不时把本地dev分支push到远程(以新建env.txt为例):
```
git add env.txt
git commit -m "add env.txt"
git push origin dev
```

若A此时也对相同文件做了修改并试图推送, 此时推送会失败, 因为A此时的提交和B已推送的提交有冲突. 解决办法: 先用`git pull`抓取origin/dev的最新提交, 在本地合并解决冲突, 再进行推送:
```
# 先指定本地dev和远程origin/dev的链接
git branch --set-upstream-to=origin/dev dev
git pull
```
pull后Git会提示产生冲突, 可以按照解决本地冲突的方法处理冲突后, 便可以把新的提交推送到远程origin/dev.

**总结**: 多人协作的工作模式一般如下:
1. 首先尝试`git push origin <branch-name>`推送自己的修改;
2. 如果失败, 说明远程分支比本地分支新, 需要`git pull`抓取远程新的分支与本地分支合并;
3. 若产生冲突, 则按照解决本地冲突的方式解决并提交到本地库;
4. 无冲突/解决冲突后, 再用`git push origin <branch-name>`推送到远程库.

> git pull 提示no tracking information, 说明本地分支和远程分支没链接起来, 因此远程库的分支不知道该合并到哪个本地分支, 因此需要使用`git branch --set-upstream-to <branch-name> origin/<branch-name>`来链接.

## 标签管理

发布一个版本时, 通常先在版本库打一个标签(tag), 来唯一确定打标签时刻的版本. 因此, 标签也就是版本库的一个快照.

更进一步, tag其实就是指向某个commit的指针. 和分支branch的区别在于: 分支可以移动以及merge, 标签不能移动. 简单来说, branch主要方便开发, tag主要方便检索(或发布)版本.

tag可理解为一个方便记忆的有意义的名字, 且和一个commit绑定. 若没有tag, 发布产品只能用commit id, 有了tag, 发布产品就能用V1.2, V2.0等有意义的名字.

**创建标签**: 首先转换到需要打tag的分支上, 然后通过`git tag <name>`建立标签:
```
git checkout master
git tag V1.0
```

**查看所有标签**:
```
git tag
```

**对过去commit创建标签**: 先找到历史commit id, 然后用`git tag <name> <commit id>`:
```
git log --pretty=oneline --abbrev-commit #查找commit id
git tag v0.9 f52c4242
```

**创建带有说明的标签**: 用`-a`指定标签名, `-m`指定说明文字:
```
git tag -a v0.1 -m "version 0.1 released"
```

**查看标签信息**:
```
git show <tag-name>
```

> tag总是和某个commit挂钩, 若该commit出现在多个分支(如master和dev), 则多个分支均可看到该tag.

### 操作标签

**删除标签**:
```
git tag -d v0.1
```

**推送标签**: tag均只存在本地, 不会自动推送到远程库. 因此需要手动推送:
```
git push origin v1.0    # 推送某个tag
git push origin --tags  # 推送所有尚未推送的tag
```

**删除已推送的远程标签**: 先从本地删除该tag, 再从远程删除:
```
git tag -d <tag-name>               # 删除本地tag
git push origin :refs/tags/v0.9     # 删除远程tag
```

## 使用GitHub

前面介绍了如何通过Git以及GitHub在本地和远程管理自己的项目, GitHub更重要的是让大家可以参与到各种开源项目的开发当中, 以bootstrap为例, 若想提交自己的代码, 可通过以下几步:

1. 去项目GitHub主页将其fork到自己的GitHub当中;
2. 通过`git clone`将自己GitHub中对应的项目克隆到本地;
3. 像往常一样在本地仓库工作, 完成后到自己的远程仓库;
4. 在GitHub上发起一个pull request, 若对方接受, 则你的code便正式进入该项目.

可以发现, 我们fork到自己GitHub中的仓库相当于项目方仓库和自己本地工作仓库的中介. 必须要从自己的GitHub中clone仓库, 如果我们直接把项目方的仓库clone到本地, 由于只有抽取而没有推送权限, 我们无法推送修改.

**断开origin远程库和本地库之间的链接**: 如果我们想本地库和另外一个远程库链接, 则要先断开和当前远程库之间的链接:
```
git remote rm origin
```

**关联多个远程库**: 将一个本地库和多个远程库关联, 需要用不同名称来标识不同远程库:
```
git remote add github git@github.com:<github-username>/<repo-name>.git
git remote add gitee git@gitee.com:<gitee-username>/<repo-name>.git
```

则两个远程库的名字分别为github何gitee, 之后再进行各种操作时应使用这两个名字而不是origin.

## 自定义Git

```
git config --global user.name <username>
git config --global user.email <email>
git config --global color.ui true
...
```

### 忽略特殊文件(.gitignore)

Git目录中有时会有一些文件不能提交到远程库, 比如python的运行临时文件, 保存密码的配置文件等等. 这样每次`git status`的时候都会显示`Untracked files ...`.

只需要在工作区根目录创建一个*.gitignore*文件, 里边包含要忽略的文件名, Git便会自动忽略这些文件.

忽略文件的原则是:

1. 忽略系统自动生成文件, 如缩略图;
2. 忽略编译生成的中间文件, 可执行文件等, 如Java编译产生的`.class`文件, python的运行文件`.pyc`;
3. 忽略带有自己敏感信息的配置文件.

**添加被忽略的文件**: 若添加一个文件到file发现无法添加, 说明该文件被`.gitignore`忽略, 可通过`-f`强制添加:

```
git add -f App.class
```

**检查.gitignore**: 若发现.gitignore忽略了不应该忽略的文件, 则可检查.gitignore, 从而发现哪条规则错误:

```
git check-ignore -v App.class # 会显示哪条规则忽略了该文件
```

> .gitignore文件本身应放到版本库里, 甚至可对其进行版本管理.

### 配置命令别名

可对git命令设置alias, 使得命令有更简洁/直观的名字. 下面是一些常见alias设置:

```
git config --global alias.<alias-name> <command>

git config --global alias.st status
git config --global alias.co checkout
git config --global alias.ci commit
git config --global alias.br branch
git config --global alias.unstage 'reset HEAD' # 撤销修改
git config --global alias.last 'log -1'        # 显示最后一次提交信息
git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
```

**配置文件**: 每个仓库均有自己的配置文件, 放置在`.git/config`当中. 而我们配置git时所加的`--global`是针对当前用户的设置(不加则只对当前repo), 该文件放置在`~/.gitconfig`中

### 搭建Git服务器

若不想公开源代码, 则需要搭建自己的Git, 过程如下:

1. 安装Git: `sudo apt-get install git`;
2. 创建一个Git用户来运行Git服务: `sudo adduser git`;
3. 创建证书登录: 收集所有需要登录用户的公钥(id_rsa.pub), 导入至`/home/git/.ssh/authorized`_keys`;
	`4. 初始化Git仓库目录: `sudo git init --bare sample.git`;
5. 把owner改为`git`: `sudo chown -R git:git sample.git`;
6. 禁止Git用户使用shell登录:
	将`/etc/passwd`中的
	```
	git:x:1001:1001:,,,:/home/git:/bin/bash
	```
	改为
	```
	git:x:1001:1001:,,,:/home/git:/usr/bin/git-shell
	```

7. 在各自电脑上克隆远程仓库: `git clone git@server:/<path>/sample.git`;
8. 之后就和GitHub交互一致了.

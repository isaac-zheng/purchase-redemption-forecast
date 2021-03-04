# purchase-redemption-forecast

## About the data

The csv data should be placed under data/ directory.

```
├── data
│   ├── comp_predict_table.csv
│   ├── mfd_bank_shibor.csv
│   ├── mfd_day_share_interest.csv
│   ├── user_balance_table.csv
│   └── user_profile_table.csv
```

## Cheat sheet for git

### Setting up git

1. [Download git](https://git-scm.com/downloads)

2. Open git bash and [set git username](https://docs.github.com/en/github/using-git/setting-your-username-in-git) and [commit email address](https://docs.github.com/en/github/setting-up-and-managing-your-github-user-account/setting-your-commit-email-address)

   ```bash
   $ git config --global user.name "yourusername"
   $ git config --global user.email "email@example.com"
   ```

   Check:

   ```bash
    git config --global user.name
    git config --global user.email
   ```

3. Authenticate with GitHub using SSH

   - Generate SSH keys

     - Open git bash

     - ```bash
       ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
       ```

     - Press enter twice

     - ```bash
       cat ~/.ssh/id_rsa.pub
       ```

     - Paste the SSH key

     - Go to your GitHub Settings - SSH and GPG keys and choose new SSH key

     - References: https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

       https://docs.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account

### Collaboration

Reference: https://medium.com/@jonathanmines/the-ultimate-github-collaboration-guide-df816e98fb67

- Clone this project and you can star it in order to find it easily in "Your stars"
  
  ```bash
  $ git clone git@github.com:xiaodianzheng/purchase-redemption-forecast.git
  ```
  
  ```bash
  $ cd purchase-redemption-forecast/
  ```
  
- Create your branch with your name or the function you will implement. Skip this step if you plan to commit to main branch.

  ```bash
  $ git checkout -b your_branch_name
  ```

  Verify:

  ```bash
  $ git branch
  ```

- If your finished your work then you can add, commit and push

  ```bash
  $ git add file_or_directory
  ```

  ```bash
  $ git commit -m "commit message"
  ```

  ```bash
  $ git push
  ```

- More useful commands can be checked [hear](https://github.com/xiaodianzheng/purchase-redemption-forecast/blob/main/Git-Cheatsheet.pdf)

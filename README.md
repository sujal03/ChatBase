# ChatBase


### MYSQL command to create DB
```bash
CREATE DATABASE UserAuthDB;

USE UserAuthDB;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL
);

CREATE TABLE data_sources (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_email VARCHAR(255),
    type VARCHAR(255),
    content TEXT
);

CREATE TABLE questions_answers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_email VARCHAR(255),
    question TEXT,
    answer TEXT
);

CREATE TABLE teams (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_email VARCHAR(255),
    team_name VARCHAR(255),
    team_url VARCHAR(255)
);
```

### Run Flask Application
```bash
python3 -m venv .venv
source .venv/bin/activate


pip install -r requirements.txt

python3 app.py
```

### Local Host Server
```bash
http://127.0.0.1:7000/
```
import os
import requests
import asyncio
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, get_flashed_messages
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from MySQLdb import IntegrityError

#  ChromaDB imports
from chroma import create_and_index_embeddings, generate_llm_response, vectordb, llm, add_text_to_vector_database, add_qa_to_vector_database, scrape_url


# Flask App Initialization
app = Flask(__name__)
app.secret_key = 'secret-key'

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PORT'] = 3306
app.config['MYSQL_PASSWORD'] = 'Root@123'
app.config['MYSQL_DB'] = 'UserAuthDB'

# Upload Folder Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# Initialize MySQL
mysql = MySQL(app)

# Utility Function: Check Allowed File Types
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# File Upload Config
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if passwords match
        if password != confirm_password:
            flash("Passwords do not match", "danger")
            return redirect(url_for('register'))

        # Hash the password
        hashed_password = generate_password_hash(password)

        try:
            cur = mysql.connection.cursor()

            # Insert the user into the database
            cur.execute(
                "INSERT INTO users (email, password) VALUES (%s, %s)", 
                (email, hashed_password)
            )
            mysql.connection.commit()
            cur.close()

            flash("You have successfully registered! Please login.", "success")
            return redirect(url_for('login'))

        except IntegrityError as e:
            if e.args[0] == 1062:  # Duplicate entry error code
                flash("Email is already registered. Please use a different email.", "danger")
            else:
                flash("An unexpected error occurred. Please try again.", "danger")
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[2], password):
            session['email'] = email
            flash("You have successfully logged in!.", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid credentials", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('email', None)
    flash("You have been logged out", "info")
    return redirect(url_for('login'))

@app.route('/request_reset', methods=['GET', 'POST'])
def request_reset():
    if request.method == 'POST':
        email = request.form['email']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()

        if user:
            session['reset_email'] = email
            return redirect(url_for('reset_password'))
        flash("Email not found", "danger")
    return render_template('request_reset.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if 'reset_email' not in session:
        return redirect(url_for('request_reset'))

    if request.method == 'POST':
        new_password = request.form['new_password']
        confirm_new_password = request.form['confirm_new_password']

        if new_password != confirm_new_password:
            flash("Passwords do not match", "danger")
            return redirect(url_for('reset_password'))

        hashed_password = generate_password_hash(new_password)
        cur = mysql.connection.cursor()
        cur.execute("UPDATE users SET password = %s WHERE email = %s", (hashed_password, session['reset_email']))
        mysql.connection.commit()
        cur.close()

        session.pop('reset_email', None)
        flash("Password reset successful", "success")
        return redirect(url_for('login'))

    return render_template('reset_password.html')

@app.route('/')
def home():
    if 'email' in session:
        return render_template('home.html', email=session['email'])
    return redirect(url_for('login'))

@app.route('/manage_data_sources', methods=['GET', 'POST'])
def manage_data_sources():
    """Manage user data sources: files, text, websites, and Q&A."""

    if 'email' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    if request.method == 'POST':

        # ✅ Handle File Upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                if not allowed_file(file.filename):
                    return jsonify({'status': 'error', 'message': "Invalid file format. Allowed formats are: txt, pdf, png, jpg, jpeg, gif"}), 400

                file_name = file.filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
                # Save the file to the upload folder
                file.save(file_path)

                # 🔹 Add file record to the database (Commented Out)
                # cur = mysql.connection.cursor()
                # cur.execute(
                #     "INSERT INTO data_sources (user_email, type, content) VALUES (%s, %s, %s)",
                #     (session['email'], 'file', file.filename)
                # )
                # mysql.connection.commit()
                # cur.close()

                # Generate embeddings for uploaded file
                item_data = {'content': file_name} # Data for UI update
                data_type = 'file'
                create_and_index_embeddings()
                flash("File uploaded and processed successfully!", "success") # Flash message
                messages = get_flashed_messages(with_categories=True) # Get flashed messages

                return jsonify({'status': 'success', 'type': data_type, 'item': item_data, 'flash_message': messages}), 200 # Include flash message in JSON


        # ✅ Handle Text Data Submission
        elif 'text' in request.form:
            text_data = request.form['text']
            if text_data:
                # Check if the text already exists in the database
                # cur = mysql.connection.cursor()
                # cur.execute(
                #     "SELECT * FROM data_sources WHERE user_email = %s AND type = %s AND content = %s",
                #     (session['email'], 'text', text_data)
                # )
                # existing_text = cur.fetchone()

                # if existing_text:
                #     flash("This text has already been added.", "warning")
                #     cur.close()
                #     return redirect(url_for('manage_data_sources'))

                # # Insert new text data
                # cur.execute(
                #     "INSERT INTO data_sources (user_email, type, content) VALUES (%s, %s, %s)",
                #     (session['email'], 'text', text_data)
                # )
                # mysql.connection.commit() 
                # cur.close()
                
                add_text_to_vector_database(text_data) # Keep this if needed in backend logic

                item_data = {'content': text_data}
                data_type = 'text'

                flash("Text data added successfully!", "success") # Flash message
                messages = get_flashed_messages(with_categories=True) # Get flashed messages
                return jsonify({'status': 'success', 'type': data_type, 'item': item_data, 'flash_message': messages}), 200 # Include flash message

        # ✅ Handle Website URL Submission
        elif 'website' in request.form:
            website_url = request.form['website']
            if website_url:
                # 🔹 Add website URL to the database (Commented Out)
                # cur = mysql.connection.cursor()
                # cur.execute(
                #     "INSERT INTO data_sources (user_email, type, content) VALUES (%s, %s, %s)",
                #     (session['email'], 'website', website_url)
                # )
                # mysql.connection.commit()
                # cur.close()
                
                # Scrape the website content (Keep scraping logic)
                prompt = "Extract relevant information"  # Customize the prompt as needed
                scraped_content = asyncio.run(scrape_url(website_url, prompt))

                message_website = 'Website URL added and content indexed successfully!' if scraped_content not in ["Crawling disallowed by robots.txt", "No content extracted"] else 'Website URL added, but content extraction failed.'
                if scraped_content not in ["Crawling disallowed by robots.txt", "No content extracted"]:
                    add_text_to_vector_database(scraped_content) # Keep vector DB update
                    flash(message_website, "success") # Flash message for success
                else:
                    flash(message_website, "warning") # Flash message for warning

                item_data = {'content': website_url}
                data_type = 'website'
                messages = get_flashed_messages(with_categories=True) # Get flashed messages for website
                return jsonify({'status': 'success', 'type': data_type, 'item': item_data, 'message': message_website, 'flash_message': messages}), 200 # Include flash message

        # ✅ Handle Q&A Submission
        elif 'question' in request.form and 'answer' in request.form:
            question = request.form['question']
            answer = request.form['answer']
            if question:
                # 🔹 Add Q&A to the database (Commented Out)
                # cur = mysql.connection.cursor()
                # cur.execute(
                #     "INSERT INTO questions_answers (user_email, question, answer) VALUES (%s, %s, %s)",
                #     (session['email'], question, answer)
                # )
                # mysql.connection.commit()
                # cur.close()

                add_qa_to_vector_database(question, answer) # Keep vector DB update

                qa_id = len(questions_answers) + 1 # Example - replace with actual ID generation (replace with real logic)
                item_data = {'question': question, 'answer': answer, 'id': qa_id, 'question_text': question, 'answer_text': answer} # Include question and answer again for UI

                data_type = 'qna'

                flash("Question and Answer submitted successfully.", "success") # Flash message
                messages = get_flashed_messages(with_categories=True) # Get flashed messages
                return jsonify({'status': 'success', 'type': data_type, 'item': item_data, 'flash_message': messages}), 200 # Include flash message

        else:
            return jsonify({'status': 'error', 'message': 'Invalid form submission'}), 400


    # ✅ Fetch User Data from Database
    cur = mysql.connection.cursor()

    # 🔹 Fetch uploaded files, text data, and websites
    cur.execute("SELECT * FROM data_sources WHERE user_email = %s", (session['email'],))
    data_sources = cur.fetchall()

    # 🔹 Fetch user-submitted Q&A
    cur.execute("SELECT * FROM questions_answers WHERE user_email = %s", (session['email'],))
    questions_answers = cur.fetchall()

    cur.close()

    return render_template('manage_data_sources.html', data_sources=data_sources, questions_answers=questions_answers)

@app.route('/delete_item/<int:item_id>', methods=['POST'])
def delete_item(item_id):
    if 'email' not in session:
        return redirect(url_for('login'))

    # Delete the item from the database
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM data_sources WHERE id = %s AND user_email = %s", (item_id, session['email']))
    mysql.connection.commit()
    cur.close()

    flash("Item deleted successfully!", "success")
    return redirect(url_for('manage_data_sources'))

@app.route('/sources', methods=['GET', 'POST'])
def sources():
    if 'email' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in
    return render_template('sources.html', email=session['email'])

@app.route('/playground', methods=['GET'])
def playground():
    if 'email' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in
    return render_template('playground.html')

@app.route('/settings', methods=['GET'])
def settings():
    if 'email' not in session:
        flash("Please log in to access settings.", "error")
        return redirect(url_for('login'))

    try:
        # Fetch user's teams from the database
        with mysql.connection.cursor() as cur:
            cur.execute("SELECT id, team_name, team_url FROM teams WHERE user_email = %s", (session['email'],))
            teams = cur.fetchall()

        # Flash a message if no teams are found
        if not teams:
            flash("You have not created any teams yet.", "info")

        return render_template('settings.html', teams=teams)

    except Exception as e:
        # Log the error for debugging
        print(f"Error fetching teams: {e}")
        flash("An error occurred while fetching your teams. Please try again later.", "error")
        return redirect(url_for('dashboard'))

@app.route('/create-team', methods=['GET', 'POST'])
def create_team():
    if 'email' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        team_name = request.form.get('team_name')
        team_url = request.form.get('team_url')

        if not team_name or not team_url:
            flash("Team name and URL cannot be empty.", "danger")
            return redirect(url_for('create_team'))

        try:
            cur = mysql.connection.cursor()
            cur.execute(
                "INSERT INTO teams (team_name, team_url, user_email) VALUES (%s, %s, %s)",
                (team_name, team_url, session['email'])
            )
            mysql.connection.commit()
            cur.close()

            flash("Team created successfully!", "success")
            return redirect(url_for('settings'))

        except IntegrityError as e:
            if e.args[0] == 1062:
                flash("A team with this name already exists.", "danger")
            else:
                flash("An unexpected error occurred. Please try again.", "danger")
            return redirect(url_for('create_team'))

    return render_template('create_team.html')

@app.route('/delete_team/<int:team_id>', methods=['GET', 'POST'])
def delete_team(team_id):
    if 'email' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM teams WHERE id = %s AND user_email = %s", (team_id, session['email']))
        mysql.connection.commit()
        cur.close()

        flash("Team deleted successfully!", "success")
        return redirect(url_for('settings'))

    # Render delete confirmation page
    cur = mysql.connection.cursor()
    cur.execute("SELECT team_name FROM teams WHERE id = %s AND user_email = %s", (team_id, session['email']))
    team = cur.fetchone()
    cur.close()

    if team:
        return render_template('delete_team.html', team_name=team[0], team_id=team_id)
    else:
        flash("Team not found.", "danger")
        return redirect(url_for('settings'))

@app.route('/update_team/<int:team_id>', methods=['GET', 'POST'])
def update_team(team_id):
    if 'email' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        team_name = request.form.get('team_name')
        team_url = request.form.get('team_url')

        if not team_name or not team_url:
            flash("Team name and URL cannot be empty.", "danger")
            return redirect(url_for('update_team', team_id=team_id))

        cur = mysql.connection.cursor()
        cur.execute(
            "UPDATE teams SET team_name = %s, team_url = %s WHERE id = %s AND user_email = %s",
            (team_name, team_url, team_id, session['email'])
        )
        mysql.connection.commit()
        cur.close()

        flash("Team updated successfully!", "success")
        return redirect(url_for('settings'))

    # Render update form
    cur = mysql.connection.cursor()
    cur.execute("SELECT team_name, team_url FROM teams WHERE id = %s AND user_email = %s", (team_id, session['email']))
    team = cur.fetchone()
    cur.close()

    if team:
        return render_template('update_team.html', team_name=team[0], team_url=team[1], team_id=team_id)
    else:
        flash("Team not found.", "danger")
        return redirect(url_for('settings'))

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if 'email' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    bot_response = ""
    user_message = ""

    if request.method == 'POST':
        user_message = request.form['user_message']

        if user_message:
            bot_response = generate_llm_response(user_message, vectordb, llm)
        else:
            bot_response = "Please enter a message."

        # ✅ Return JSON response for AJAX requests
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"bot_response": bot_response})

    # Regular page render for direct GET requests
    return render_template('chatbot.html', email=session['email'], user_message=user_message, bot_response=bot_response)

if __name__ == '__main__':
    app.run(debug=True, port=7000, host='0.0.0.0')

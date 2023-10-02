from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify, flash
from flask_mysqldb import MySQL, MySQLdb
import mysql.connector
import bcrypt 
import werkzeug
import cv2
from PIL import Image
import numpy as np
import os
import time
from datetime import date
 
app = Flask(__name__)
 
cnt = 0
pause_cnt = 0
justscanned = False

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="duamei1"
)
mycursor = mydb.cursor()

app.secret_key = "membuatLOginFlask1"
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'duamei1'
sql = MySQL(app)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    face_classifier = cv2.CascadeClassifier("C:/Users/Lenovo/Documents/projectku/resources/haarcascade_frontalface_default.xml")
 
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.2, 5)

        if faces == ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face
 
    cap = cv2.VideoCapture(1)
 
    mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]
    img_id = lastid
    max_imgid = img_id + 10
    count_img = 0
 
    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = "dataset/"+nbr+"."+ str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
 
            mycursor.execute("""INSERT INTO `img_dataset` (`img_id`, `img_person`) VALUES
                                ('{}', '{}')""".format(img_id, nbr))
            mydb.commit()
 
            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            #13 itu adalah kode untuk tombol enter
            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                break
                cap.release()
                cv2.destroyAllWindows()
 
 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = "C:/Users/Lenovo/Documents/projectku/dataset"
 
    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []
 
    for image in path:
        img = Image.open(image).convert('L');
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
 
        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)
 
    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
 
    return redirect('/datasiswa')
 
 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition():  # generate frame by frame from camera
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
 
        global justscanned
        global pause_cnt
 
        pause_cnt += 1
        coords = []
 
        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            if confidence > 70 and not justscanned:
                global cnt
                cnt += 1
                n = (100 / 30) * cnt
                w_filled = (cnt / 30) * w
                
                cv2.putText(img, str(int(n))+' %', (x + 20, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), color, 2)
                cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled), y + h + 50), (153, 255, 255), cv2.FILLED)
                
                mycursor.execute("select a.img_person, b.prs_name, b.kelas "
                                 "  from img_dataset a "
                                 "  left join siswa b on a.img_person = b.prs_nbr "
                                 " where img_id = " + str(id))
                row = mycursor.fetchone()
                pnbr = row[0]
                pname = row[1]
                pskill = row[2]
                cv2.putText(img, pname + ' | ' + pskill, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
                
                if int(cnt) == 30:
                    cnt = 0
                    mycursor.execute("insert into absen (accs_date, accs_prsn) values('"+str(date.today())+"', '" + pnbr + "')")
                    mydb.commit()
 
                    cv2.putText(img, pname + ' | ' + pskill, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
                    time.sleep(1)
                    justscanned = True
                    pause_cnt = 0
 
            else:
                if not justscanned:
                    cv2.putText(img, 'UNKNOWN', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,cv2.LINE_AA)
 
                if pause_cnt > 80:
                    justscanned = False
 
            coords = [x, y, w, h]
        return coords
 
    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 8, (255, 255, 0), "Face", clf)
        return img
 
    faceCascade = cv2.CascadeClassifier("C:/Users/Lenovo/Documents/projectku/resources/haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
 
    wCam, hCam = 400, 400
    cap = cv2.VideoCapture(1)
    cap.set(3, wCam)
    cap.set(4, hCam)
 
    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
 
        key = cv2.waitKey(1)
        if key == 27:
            break
 
@app.route('/')
def home():
    return render_template('coba.html')
#untuk login
@app.route('/login', methods=['GET', 'POST'])
def login(): 
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password'].encode('utf-8')
        curl = sql.connection.cursor(MySQLdb.cursors.DictCursor)
        curl.execute("SELECT * FROM pengguna WHERE email=%s",(email,))
        user = curl.fetchone()
        curl.close()

        if user is not None and len(user) > 0 :
            if bcrypt.hashpw(password, user['password'].encode('utf-8')) == user['password'].encode('utf-8'):
                session['username'] = user ['username']
                session['email'] = user['email']
                session['role'] = user['role']
                return redirect(url_for('home'))
            else :
                flash("Gagal, Email dan Password Tidak Cocok")
                return redirect(url_for('login'))
        else :
            flash("Gagal, User Tidak Ditemukan")
            return redirect(url_for('login'))
    else: 
        return render_template("login.html")

#untuk regis akun baru
@app.route('/register', methods=['POST', 'GET'])
def register():
    if 'role' in session:
        if request.method=='GET':
            return render_template('register.html')
        else :
            username = request.form['username']
            email = request.form['email']
            role = request.form['role']
            password = request.form['password'].encode('utf-8')
            hash_password = bcrypt.hashpw(password, bcrypt.gensalt())

            cur = sql.connection.cursor()
            cur.execute("INSERT INTO pengguna (username,email,role,password) VALUES (%s,%s,%s,%s)" ,(username, email, role, hash_password)) 
            sql.connection.commit()
            flash("Berhasil membuat akun!")
            return redirect(url_for('register'))
    else:
        flash("Anda belum login, silahkan login dulu!")
        return redirect(url_for('login'))
    
    
@app.route('/updatedatasiswa', methods=["GET", "POST"])
def updatedatasiswa():
    if request.method == 'POST':
        prs_nbr = request.form['prs_nbr']
        prs_name = request.form['prs_name']
        kelas = request.form['kelas']
        kelamin = request.form['kelamin']
        prs_added = request.form['prs_added']
        mycursor.execute("""UPDATE siswa SET prs_name=%s, kelas=%s, kelamin=%s, prs_added=%s WHERE prs_nbr=%s""", (prs_name, kelas, kelamin, prs_added, prs_nbr))
        mydb.commit()
        return redirect(url_for('datasiswa'))
#rute ke halaman data siswa 
@app.route('/datasiswa')
def datasiswa():
    if 'role' in session:
        mycursor.execute("select prs_nbr, prs_name, kelas, kelamin, prs_added from siswa order by 1 desc")
        data = mycursor.fetchall()
        return render_template('datasiswa.html', data=data)
    else:
        flash("Anda belum login, silahkan login dulu!")
        return redirect(url_for('login'))
#tombol delete siswa
@app.route('/delete/<prs_nbr>', methods=['GET'])
def deletesiswa(prs_nbr):
    if request.method == 'GET':
        mycursor.execute('''
        DELETE 
        FROM siswa 
        WHERE prs_nbr=%s''', (prs_nbr, ))
        mydb.commit()

        return redirect(url_for('datasiswa'))

#rute halaman tambah data
@app.route('/addprsn')
def addprsn():
    mycursor.execute("select ifnull(max(prs_nbr) + 1, 101) from siswa")
    row = mycursor.fetchone()
    nbr = row[0]
    # print(int(nbr))
 
    return render_template('tambahdata.html', newnbr=int(nbr))

#untuk tombol sumbit tambah data
@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    prsnbr = request.form.get('prs_nbr')
    prsname = request.form.get('prs_name')
    kelas = request.form.get('kelas')
    kelamin = request.form.get('kelamin')
 
    mycursor.execute("""INSERT INTO `siswa` (`prs_nbr`, `prs_name`, `kelas`, `kelamin`) VALUES
                    ('{}', '{}', '{}', '{}')""".format(prsnbr, prsname, kelas, kelamin))
    mydb.commit()
 
    # return redirect(url_for('home'))
    return redirect(url_for('vfdataset_page', prs=prsnbr))

#rute sehabis tekan tombol sumbit langsung menangkap foto muka  
@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('gendataset.html', prs=prs)


@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')
 
 
@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')
 
@app.route('/fr_page')
def fr_page():
    """Video streaming home page."""
    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.kelas, a.accs_added "
                     "  from absen a "
                     "  left join siswa b on a.accs_prsn = b.prs_nbr "
                     " where a.accs_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()
 
    return render_template('fr_page.html', data=data)
 
 
@app.route('/countTodayScan')
def countTodayScan():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="2mei"
    )
    mycursor = mydb.cursor()
 
    mycursor.execute("select count(*) "
                     "  from absen "
                     " where accs_date = curdate() ")
    row = mycursor.fetchone()
    rowcount = row[0]
 
    return jsonify({'rowcount': rowcount})
 
 
@app.route('/loadData', methods = ['GET', 'POST'])
def loadData():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="2mei"
    )
    mycursor = mydb.cursor()
 
    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.kelas, date_format(a.accs_added, '%H:%i:%s') "
                     "  from absen a "
                     "  left join siswa b on a.accs_prsn = b.prs_nbr "
                     " where a.accs_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()
 
    return jsonify(response = data)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< absen X ipa 1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/dataabsen_xipa1')
def dataabsen_ipa1():
    if 'role' in session:
        mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.kelas, date_format(a.accs_added, '%e-%m-%Y,%H:%i:%s') "
                        "  from absen a "
                        "  left join siswa b on a.accs_prsn = b.prs_nbr "
                        " where b.kelas = 'X-IPA1' "
                        " order by 1 desc")
        data = mycursor.fetchall()
    
        return render_template('dataabsen_ipa1.html', data=data)
    else:
        flash("Anda belum login, silahkan login dulu!")
        return redirect(url_for('login'))

@app.route('/dataabsen_xips1')
def dataabsen_ips1():
    if 'role' in session:
        mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.kelas, date_format(a.accs_added, '%e-%m-%Y,%H:%i:%s') "
                        "  from absen a "
                        "  left join siswa b on a.accs_prsn = b.prs_nbr "
                        " where b.kelas = 'X-IPS1' "
                        " order by 1 desc")
        data = mycursor.fetchall()
    
        return render_template('dataabsen_ips1.html', data=data)
    else:
        flash("Anda belum login, silahkan login dulu!")
        return redirect(url_for('login'))
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< absen XI>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/dataabsen_xiips1')
def dataabsen_xiips1():
    if 'role' in session:
        mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.kelas, date_format(a.accs_added, '%e-%m-%Y,%H:%i:%s') "
                        "  from absen a "
                        "  left join siswa b on a.accs_prsn = b.prs_nbr "
                        " where b.kelas = 'XI-IPS1' "
                        " order by 1 desc")
        data = mycursor.fetchall()
    
        return render_template('dataabsen_xiips1.html', data=data)
    else:
        flash("Anda belum login, silahkan login dulu!")
        return redirect(url_for('login'))
    
@app.route('/dataabsen_xiipa1')
def dataabsen_xiipa1():
    if 'role' in session:
        mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.kelas, date_format(a.accs_added, '%e-%m-%Y,%H:%i:%s') "
                        "  from absen a "
                        "  left join siswa b on a.accs_prsn = b.prs_nbr "
                        " where b.kelas = 'XI-IPA1' "
                        " order by 1 desc")
        data = mycursor.fetchall()
    
        return render_template('dataabsen_xiips1.html', data=data)
    else:
        flash("Anda belum login, silahkan login dulu!")
        return redirect(url_for('login'))

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< absen XII>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/dataabsen_xiiips1')
def dataabsen_xiiips1():
    if 'role' in session:
        mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.kelas, date_format(a.accs_added, '%e-%m-%Y,%H:%i:%s') "
                        "  from absen a "
                        "  left join siswa b on a.accs_prsn = b.prs_nbr "
                        " where b.kelas = 'XII-IPS1' "
                        " order by 1 desc")
        data = mycursor.fetchall()
    
        return render_template('dataabsen_xiiips1.html', data=data)
    else:
        flash("Anda belum login, silahkan login dulu!")
        return redirect(url_for('login'))
    
@app.route('/dataabsen_xiiipa1')
def dataabsen_xiiipa1():
    if 'role' in session:
        mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.kelas, date_format(a.accs_added, '%e-%m-%Y,%H:%i:%s') "
                        "  from absen a "
                        "  left join siswa b on a.accs_prsn = b.prs_nbr "
                        " where b.kelas = 'XII-IPA1' "
                        " order by 1 desc")
        data = mycursor.fetchall()
    
        return render_template('dataabsen_xiiips1.html', data=data)
    else:
        flash("Anda belum login, silahkan login dulu!")
        return redirect(url_for('login'))
#untuk logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home')) 

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)

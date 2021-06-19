import os

from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('jobs.pkl', 'rb'))

model2 = pickle.load(open('startup.pkl', 'rb'))

model3 = pickle.load(open('salary_model.pkl', 'rb'))

city_dict={'San Francisco': 128,
 'New York': 91,
 'Mountain View': 47,
 'Palo Alto': 35,
 'Santa Clara': 27,
 'Austin': 27,
 'San Mateo': 26,
 'Seattle': 26,
 'Sunnyvale': 22,
 'San Jose': 18,
 'Cambridge': 16,
 'Menlo Park': 16,
 'San Diego': 15,
 'Los Angeles': 15,
 'Redwood City': 14,
 'Chicago': 13,
 'Boston': 13,
 'Waltham': 12,
 'Burlington': 11,
 'Santa Monica': 10,
 'Campbell': 9,
 'Cupertino': 8,
 'Fremont': 8,
 'Bellevue': 7,
 'Atlanta': 7,
 'Boulder': 6,
 'Brooklyn': 5,
 'Burlingame': 5,
 'Portland': 5,
 'Pasadena': 4,
 'Foster City': 4,
 'San Bruno': 4,
 'Washington': 4,
 'Addison': 4,
 'Pittsburgh': 4,
 'Tampa': 4,
 'Richardson': 4,
 'Carlsbad': 3,
 'Lexington': 3,
 'New York City': 3,
 'Berkeley': 3,
 'Milpitas': 3,
 'Durham': 3,
 'Pleasanton': 3,
 'McLean': 3,
 'Providence': 3,
 'Englewood': 3,
 'Raleigh': 3,
 'Boxborough': 3,
 'Denver': 3,
 'Irvine': 3,
 'Los Gatos': 3,
 'South San Francisco': 3,
 'Woburn': 3,
 'Cincinnati': 3,
 'Philadelphia': 3,
 'Petaluma': 2,
 'Calabasas': 2,
 'La Jolla': 2,
 'Alameda': 2,
 'NY': 2,
 'Louisville': 2,
 'El Segundo': 2,
 'Hollywood': 2,
 'Santa Barbara': 2,
 'Allentown': 2,
 'San Rafael': 2,
 'Bothell': 2,
 'Aliso Viejo': 2,
 'Redmond': 2,
 'Los Altos': 2,
 'Kirkland': 2,
 'Indianapolis': 2,
 'Acton': 2,
 'Dallas': 2,
 'Houston': 2,
 'Emeryville': 2,
 'Nashville': 2,
 'Billerica': 2,
 'Needham': 2,
 'Marlborough': 2,
 'Brisbane': 2,
 'Tempe': 2,
 'Yorba Linda': 1,
 'Pittsboro': 1,
 'Freedom': 1,
 'Golden Valley': 1,
 'Somerset': 1,
 'Bala Cynwyd': 1,
 'Thousand Oaks': 1,
 'Potomac Falls': 1,
 'North Reading': 1,
 'Timonium': 1,
 'Viena': 1,
 'Jersey City': 1,
 'Lisle': 1,
 'Henderson': 1,
 'Paramus': 1,
 'Kansas City': 1,
 'Larkspur': 1,
 'Dedham': 1,
 'El Segundo,': 1,
 'Salt Lake City': 1,
 'Champaign': 1,
 'Playa Vista': 1,
 'Waco': 1,
 'Minnetonka': 1,
 'Frederick': 1,
 'San Franciso': 1,
 'Vancouver': 1,
 'Cleveland': 1,
 'Hillsborough': 1,
 'Nashua': 1,
 'Provo': 1,
 'Canton': 1,
 'Bethesda': 1,
 'Toledo': 1,
 'Santa Ana': 1,
 'Tualatin': 1,
 'Woodbury': 1,
 'Manchester': 1,
 'Evanston': 1,
 'West Chester': 1,
 'Sterling': 1,
 'Duluth': 1,
 'Lake Oswego': 1,
 'Westport': 1,
 'Westford': 1,
 'Columbia': 1,
 'Rye Brook': 1,
 'Newport Beach': 1,
 'The Woodlands': 1,
 'Bloomfield': 1,
 'Hampton': 1,
 'Torrance': 1,
 'Belmont': 1,
 'Little Rock': 1,
 'Middleton': 1,
 'Moffett Field': 1,
 'Altamonte Springs': 1,
 'Dulles': 1,
 'Minneapolis': 1,
 'Laguna Niguel': 1,
 'Long Island City': 1,
 'Maynard': 1,
 'Idaho Falls': 1,
 'Centennial': 1,
 'Alpharetta': 1,
 'Arlington': 1,
 'North Billerica': 1,
 'North Hollywood': 1,
 'New Hope': 1,
 'College Park': 1,
 'Longmont': 1,
 'Arcadia': 1,
 'Plymouth': 1,
 'Las Vegas': 1,
 'Chelmsford': 1,
 'Wilmington': 1,
 'Napa': 1,
 'Bethlehem': 1,
 'Andover': 1,
 'Morgan Hill': 1,
 'Hartford': 1,
 'Yardley': 1,
 'Naperville': 1,
 'Farmington': 1,
 'Sunnnyvale': 1,
 'Carpinteria': 1,
 'Newton': 1,
 'Annapolis': 1,
 'Warrenville': 1,
 'Columbus': 1,
 'Lindon': 1,
 'Chantilly': 1,
 'Loveland': 1,
 'Red Bank': 1,
 'Albuquerque': 1,
 'Glendale': 1,
 'West Newfield': 1,
 'West Hollywood': 1,
 'Berwyn': 1,
 'Monterey Park': 1,
 'Lowell': 1,
 'Zeeland': 1,
 'Lawrenceville': 1,
 'Chevy Chase': 1,
 'Solana Beach': 1,
 'Greenwood Village': 1,
 'Itasca': 1,
 'Oakland': 1,
 'Conshohocken': 1,
 'NW Atlanta': 1,
 'Saint Paul': 1,
 'Plano': 1,
 'Princeton': 1,
 'Somerville': 1,
 'Charlottesville': 1,
 'Puyallup': 1,
 'Saint Louis': 1,
 'Beverly Hills': 1,
 'Littleton': 1,
 'San Carlos': 1,
 'Weston': 1,
 'NYC': 1,
 'Bingham Farms': 1,
 'Reston': 1,
 'Scotts Valley': 1,
 'Tewksbury': 1,
 'Framingham': 1,
 'Broomfield': 1,
 'Lancaster': 1,
 'Williamstown': 1,
 'Herndon': 1,
 'Memphis': 1,
 'Avon': 1,
 'SPOKANE': 1,
 'Kenmore': 1,
 'Kearneysville': 1,
 'Vienna': 1,
 'Bedford': 1}
@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()


@app.route("/predict", methods=['POST','GET'])
def predict():
    Fuel_Type_Diesel = 0
    if request.method == 'POST':
        salary_range = request.form['salary']
        if salary_range=='yes':
            salary_range=1
        else:
            salary_range=0

        work_from_home = request.form['wfh']
        if work_from_home == 'yes':
            work_from_home = 1
        else:
            work_from_home = 0

        logo = request.form['logo']
        if logo == 'yes':
            logo = 1
        else:
            logo = 0

        questions = request.form['question']
        if questions == 'yes':
            questions = 1
        else:
            questions = 0

        company_profile = request.form['profile']
        if company_profile == 'yes':
            company_profile = 1
        else:
            company_profile = 0

        requirements_present = request.form['req']
        if requirements_present == 'yes':
            requirements_present = 1
        else:
            requirements_present = 0

        benefits = request.form['benefits']
        if benefits == 'yes':
            benefits = 1
        else:
            benefits = 0


        country=request.form['country']
        country_arr=[]
        if (country == 'USA'):
            country_arr=[1,0,0,0,0,0,0,0,0,0]
        elif(country == 'UK'):
            country_arr=[0,1,0,0,0,0,0,0,0,0]
        elif(country=='Greece'):
            country_arr=[0,0,1,0,0,0,0,0,0,0]
        elif (country == 'Canada'):
            country_arr=[0,0,0,1,0,0,0,0,0,0]
        elif (country == 'Germany'):
            country_arr=[0,0,0,0,1,0,0,0,0,0]
        elif (country == 'Not Given'):
            country_arr=[0,0,0,0,0,1,0,0,0,0]
        elif (country == 'New Zealand'):
            country_arr=[0,0,0,0,0,0,1,0,0,0]
        elif (country == 'India'):
            country_arr=[0,0,0,0,0,0,0,1,0,0]
        elif (country == 'Australia'):
            country_arr=[0,0,0,0,0,0,0,0,1,0]
        elif (country == 'Philippines'):
            country_arr=[0,0,0,0,0,0,0,0,0,1]

        else:
            country_arr=[0,0,0,0,0,0,0,0,0,0]

        department = request.form['dept']
        dept_arr = []
        if (department == 'not given'):
            dept_arr = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif (department == 'Sales'):
            dept_arr = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif (department == 'Engineering'):
            dept_arr = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif (department == 'Marketing'):
            dept_arr = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif (department == 'Operations'):
            dept_arr = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif (department == 'IT'):
            dept_arr = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif (department == 'Development'):
            dept_arr = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif (department == 'Product'):
            dept_arr = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif (department == 'Information Technology'):
            dept_arr = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif (department == 'Design'):
            dept_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        else:
            dept_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        qualification = request.form['qualification']
        qualification_arr = []
        if (qualification == 'not given'):
            qualification_arr = [1, 0, 0, 0, 0]
        elif (qualification == "Bachelor's Degree"):
            qualification_arr = [0, 1, 0, 0, 0]
        elif (qualification == 'High School or equivalent'):
            qualification_arr = [0, 0, 1, 0, 0]
        elif (qualification == "Master's Degree"):
            qualification_arr = [0, 0, 0, 1, 0]
        elif (qualification == 'Associate Degree'):
            qualification_arr = [0, 0, 0, 0, 1]

        else:
            qualification_arr = [0, 0, 0, 0, 0]

        industry = request.form['industry']
        industry_arr = []
        if (industry == 'not given'):
            industry_arr = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif (industry == 'Information Technology and Services'):
            industry_arr = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif (industry == 'Computer Software'):
            industry_arr = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif (industry == 'Internet'):
            industry_arr = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif (industry == 'Marketing and Advertising'):
            industry_arr = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif (industry == 'Education Management'):
            industry_arr = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif (industry == 'Financial Services'):
            industry_arr = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif (industry == 'Hospital & Health Care'):
            industry_arr = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif (industry == 'Consumer Services'):
            industry_arr = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif (industry == 'Telecommunications'):
            industry_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        else:
            industry_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        function = request.form['function']
        function_arr = []
        if (function == 'not given'):
            function_arr = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif (function == 'Information Technology'):
            function_arr = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif (function == 'Sales'):
            function_arr = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif (function == 'Engineering'):
            function_arr = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif (function == 'Customer Service'):
            function_arr = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif (function == 'Marketing'):
            function_arr = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif (function == 'Administrative'):
            function_arr = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif (function == 'Design'):
            function_arr = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif (function == 'Health Care Provider'):
            function_arr = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif (function == 'Education'):
            function_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        else:
            function_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        experience = request.form['experience']
        experience_arr = []
        if (experience == 'Director'):
            experience_arr = [1, 0, 0, 0, 0, 0, 0]
        elif (experience == 'Entry level'):
            experience_arr = [0, 1, 0, 0, 0, 0, 0]
        elif (experience == 'Executive'):
            experience_arr = [0, 0, 1, 0, 0, 0, 0]
        elif (experience == 'Internship'):
            experience_arr = [0, 0, 0, 1, 0, 0, 0]
        elif (experience == 'Mid-Senior level'):
            experience_arr = [0, 0, 0, 0, 1, 0, 0]
        elif (experience == 'Not Applicable'):
            experience_arr = [0, 0, 0, 0, 0, 1, 0]
        elif (experience == 'not given'):
            experience_arr = [0, 0, 0, 0, 0, 0, 1]


        else:
            experience_arr = [0, 0, 0, 0, 0, 0, 0]

        emp_type = request.form['emp_type']
        emp_type_arr = []
        if (emp_type == 'Full-time'):
            emp_type_arr = [1, 0, 0, 0, 0]
        elif (emp_type == 'Other'):
            emp_type_arr = [0, 1, 0, 0, 0]
        elif (emp_type == 'Part-time'):
            emp_type_arr = [0, 0, 1, 0, 0]
        elif (emp_type == 'Temporary'):
            emp_type_arr = [0, 0, 0, 1, 0]
        elif (emp_type == 'not given'):
            emp_type_arr = [0, 0, 0, 0, 1]


        featueres=[salary_range,work_from_home,logo,questions,company_profile,requirements_present,benefits]+\
                  country_arr+dept_arr+qualification_arr+industry_arr+function_arr+experience_arr+emp_type_arr
        prediction = model.predict([featueres])
        output = prediction[0]
        op_dict={0:'Real',1:'Fake'}
        op=op_dict[output]
        flag=True
        if op=='Real':
            flag=False
        print(output)
        return render_template('home.html', prediction_texts="The job posting is   {}".format(op),flag=flag)

    else:
        return render_template('home.html')

@app.route("/startup_success", methods=['POST','GET'])
def predict_success():
    if request.method == 'POST':
        city = request.form['city']
        city=city_dict[city]

        foundation_yr= 2021-int(request.form['foundation'])

        first_fund_age=int(request.form['fund1'])
        last_fund_age = int(request.form['fund2'])

        first_milestone_age = int(request.form['milestone1'])
        last_milestone_age = int(request.form['milestone1'])

        investors=int(request.form['investors'])

        rounds = int(request.form['rounds'])
        funding = int(request.form['funding'])

        milestones=int(request.form['milestones'])

        vc=request.form['vc']
        if vc=='yes':
            vc=1
        else:
            vc=0

        angel = request.form['angel']
        if angel == 'yes':
            angel = 1
        else:
            angel = 0

        top = request.form['top']
        if top == 'yes':
            top = 1
        else:
            top = 0

        participants=request.form['participants']



        state=request.form['state']
        state_arr=[]
        if (state == 'ca'):
            state_arr=[1,0,0,0,0,0,0,0,0,0]
        elif(state == 'ny'):
            state_arr=[0,1,0,0,0,0,0,0,0,0]
        elif(state=='ma'):
            state_arr=[0,0,1,0,0,0,0,0,0,0]
        elif (state == 'tx'):
            state_arr=[0,0,0,1,0,0,0,0,0,0]
        elif (state == 'wa'):
            state_arr=[0,0,0,0,1,0,0,0,0,0]
        elif (state == 'co'):
            state_arr=[0,0,0,0,0,1,0,0,0,0]
        elif (state == 'il'):
            state_arr=[0,0,0,0,0,0,1,0,0,0]
        elif (state == 'pa'):
            state_arr=[0,0,0,0,0,0,0,1,0,0]
        elif (state == 'va'):
            state_arr=[0,0,0,0,0,0,0,0,1,0]
        elif (state == 'ga'):
            state_arr=[0,0,0,0,0,0,0,0,0,1]

        else:
            state_arr=[0,0,0,0,0,0,0,0,0,0]

        category = request.form['category']
        category_arr = []
        if (category == 'software'):
            category_arr = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif (category == 'web'):
            category_arr = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif (category == 'mobile'):
            category_arr = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif (category == 'enterprise'):
            category_arr = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif (category == 'advertising'):
            category_arr = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif (category == 'games_video'):
            category_arr = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif (category == 'semiconductor'):
            category_arr = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif (category == 'biotech'):
            category_arr = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif (category == 'network_hosting'):
            category_arr = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif (category == 'hardware'):
            category_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        else:
            category_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]



        featueres=[city,foundation_yr,first_fund_age,last_fund_age,first_milestone_age,last_milestone_age,investors,rounds,funding,milestones,vc,angel,participants,top]+\
                  state_arr+category_arr
        prediction = model2.predict([featueres])
        output = prediction[0]
        op_dict={0:'Fail',1:'Succeed'}
        op=op_dict[output]

        print(output)
        return render_template('startup.html', prediction_texts="The startup will {}".format(op))

    else:
        return render_template('startup.html')


@app.route("/salary", methods=['POST','GET'])
def predict_salary():



    if request.method == 'POST':
        per10 = float(request.form['10th'])
        per12 = float(request.form['12th'])
        grad_per = float(request.form['grad'])

        college_tier = int(request.form['tier'])
        grad_yr = 2021 - int(request.form['gradyr'])

        english = int(request.form['english'])
        logical = int(request.form['logical'])
        quant = int(request.form['quant'])

        Domain = float(request.form['domain'])
        Programming = int(request.form['cp'])
        electronics = int(request.form['ent'])

        conscientiousness = float(request.form['p1'])
        agreeableness = float(request.form['p2'])
        extraversion = float(request.form['p3'])
        nueroticism = float(request.form['p4'])
        openess_to_experience = float(request.form['p5'])

        age=int(request.form['age'])

        board_10=request.form['board1']
        board_10_arr=[]
        if board_10=='icse':
            board_10_arr=[1,0]
        elif board_10=='ssc':
            board_10_arr=[0,1]
        else:
            board_10_arr=[0,0]

        board_12 = request.form['board2']
        board_12_arr = []
        if board_12 == 'cbse':
            board_12_arr = [1, 0,0,0,0]
        elif board_12 == 'state_board':
            board_12_arr = [0, 1,0,0,0]
        elif board_12 == 'icse':
            board_12_arr = [0, 0,0,1,0]
        elif board_12 == 'up':
            board_12_arr = [0, 0,0,0,1]
        else:
            board_12_arr = [0, 0,0,0,0]

        stream = request.form['stream']
        stream_arr=[]
        if (stream == 'electronics and communication engineering'):
            stream_arr=[1,0,0,0,0,0,0,0,0]
        elif(stream == 'computer science & engineering'):
            stream_arr=[0,1,0,0,0,0,0,0,0]
        elif(stream=='information technology'):
            stream_arr=[0,0,1,0,0,0,0,0,0]
        elif (stream == 'computer engineering'):
            stream_arr=[0,0,0,1,0,0,0,0,0]
        elif (stream == 'mechanical engineering'):
            stream_arr=[0,0,0,0,1,0,0,0,0]
        elif (stream == 'electronics and electrical engineering'):
            stream_arr=[0,0,0,0,0,1,0,0,0]
        elif (stream == 'electronics & telecommunications'):
            stream_arr=[0,0,0,0,0,0,1,0,0]
        elif (stream == 'electrical engineering'):
            stream_arr=[0,0,0,0,0,0,0,1,0]
        elif (stream == 'electronics & instrumentation eng'):
            stream_arr=[0,0,0,0,0,0,0,0,1]


        else:
            stream_arr=[0,0,0,0,0,0,0,0,0,0]

        state = request.form['state']
        state_arr = []
        if (state == 'Uttar Pradesh'):
            state_arr = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif (state == 'Karnataka'):
            state_arr = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif (state == 'Tamil Nadu'):
            state_arr = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif (state == 'Telangana'):
            state_arr = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif (state == 'Maharashtra'):
            state_arr = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif (state == 'Andhra Pradesh'):
            state_arr = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif (state == 'West Bengal'):
            state_arr = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif (state == 'Madhya Pradesh'):
            state_arr = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif (state == 'Punjab'):
            state_arr = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif (state == 'Haryana'):
            state_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        else:
            state_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        gender=request.form['gender']

        degree = request.form['degree']
        degree_arr = []
        if (degree == 'M.Sc. (Tech.)'):
            degree_arr = [1, 0, 0]
        elif (degree == "M.Tech./M.E."):
            degree_arr = [0, 1, 0]
        elif (degree == 'MCA'):
            degree_arr = [0, 0, 1]

        else:
            degree_arr = [0, 0, 0]




        featueres=[per10,per12,grad_per,college_tier,grad_yr,english,logical,quant,Domain,Programming,electronics,conscientiousness,agreeableness,extraversion,nueroticism,openess_to_experience,age]+\
                  board_10_arr+board_12_arr+stream_arr+state_arr+[gender]+degree_arr
        prediction = model3.predict([featueres])
        output = round(prediction[0],0)

        return render_template('salary.html', prediction_texts="Predicted salary is   {}".format(output))

    else:
        return render_template('salary.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)


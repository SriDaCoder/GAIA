import json
import os
import random
import streamlit as st

JSON_FILE = json.load(open("APOLLO/topics_data.json", "r"))

"""
Subjects:
1. Physics
2. Math
3. Chemistry
4. Biology
5. Computer Science
"""

def generate_quiz():
    st.button("Physics", on_click=physics)
    st.button("Math", on_click=math)
    st.button("Chemistry", on_click=chemistry)
    st.button("Biology", on_click=biology)

def physics():
    st.write("What topic in physics?")
    for topic in JSON_FILE["physics"]:
        st.button(topic, on_click=physics_topic, args=(topic,))
    st.button("Back", on_click=generate_quiz)

def math():
    st.write("What topic in math?")
    for topic in JSON_FILE["math"]:
        st.button(topic, on_click=math_topic, args=(topic,))
    st.button("Back", on_click=generate_quiz)

def chemistry():
    st.write("What topic in chemistry?")
    for topic in JSON_FILE["chemistry"]:
        st.button(topic, on_click=chemistry_topic, args=(topic,))
    st.button("Back", on_click=generate_quiz)

def biology():
    st.write("What topic in biology?")
    for topic in JSON_FILE["biology"]:
        st.button(topic, on_click=biology_topic, args=(topic,))
    st.button("Back", on_click=generate_quiz)

def physics_topic(topic):
    st.write(f"Generating quiz for {topic} in physics...")
    # Here you would call the function to generate the quiz
    # For now, we'll just simulate it with a random number of questions
    num_questions = random.randint(1, 10)
    st.write(f"Generated {num_questions} questions for {topic} in physics.")
    st.button("Back", on_click=physics)

def math_topic(topic):
    st.write(f"Generating quiz for {topic} in math...")
    # Here you would call the function to generate the quiz
    # For now, we'll just simulate it with a random number of questions
    num_questions = random.randint(1, 10)
    st.write(f"Generated {num_questions} questions for {topic} in math.")
    st.button("Back", on_click=math)

def chemistry_topic(topic):
    st.write(f"Generating quiz for {topic} in chemistry...")
    # Here you would call the function to generate the quiz
    # For now, we'll just simulate it with a random number of questions
    num_questions = random.randint(1, 10)
    st.write(f"Generated {num_questions} questions for {topic} in chemistry.")
    st.button("Back", on_click=chemistry)

def biology_topic(topic):
    st.write(f"Generating quiz for {topic} in biology...")
    # Here you would call the function to generate the quiz
    # For now, we'll just simulate it with a random number of questions
    num_questions = random.randint(1, 10)
    st.write(f"Generated {num_questions} questions for {topic} in biology.")
    st.button("Back", on_click=biology)

st.title("APOLLO")

st.write("APOLLO is a tool for teaching people of all ages about all types of subjects.")
st.write("It comes with the following features:")
st.write("- Quiz generation")
st.write("- Study material")

st.button("Generate Quiz", on_click=generate_quiz)
# st.button("Generate Study Material", key="generate_study_material")
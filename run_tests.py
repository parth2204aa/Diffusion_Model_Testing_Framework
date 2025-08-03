### run_tests.py
from prompts import PROMPTS
from model_inference import load_model, generate_images
from report_generator import generate_report

pipe = load_model()
results = generate_images(pipe, PROMPTS)
generate_report(results)

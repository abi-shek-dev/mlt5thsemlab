from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

model = DiscreteBayesianNetwork([
    ('Fever', 'COVID'),
    ('Cough', 'COVID'),
    ('TravelHistory', 'COVID'),
    ('ContactWithInfected', 'COVID'),
    ('COVID', 'TestResult')
])

cpd_fever = TabularCPD(variable='Fever', variable_card=2, values=[[0.7], [0.3]])
cpd_cough = TabularCPD(variable='Cough', variable_card=2, values=[[0.6], [0.4]])
cpd_travel = TabularCPD(variable='TravelHistory', variable_card=2, values=[[0.85], [0.15]])
cpd_contact = TabularCPD(variable='ContactWithInfected', variable_card=2, values=[[0.9], [0.1]])

cpd_covid = TabularCPD(
    variable='COVID', variable_card=2,
    values=[
        [0.99, 0.95, 0.97, 0.90, 0.92, 0.85, 0.87, 0.75,
         0.80, 0.60, 0.65, 0.50, 0.40, 0.30, 0.20, 0.10],
        [0.01, 0.05, 0.03, 0.10, 0.08, 0.15, 0.13, 0.25,
         0.20, 0.40, 0.35, 0.50, 0.60, 0.70, 0.80, 0.90],
    ],
    evidence=['Fever', 'Cough', 'TravelHistory', 'ContactWithInfected'],
    evidence_card=[2, 2, 2, 2]
)

cpd_test_result = TabularCPD(
    variable='TestResult',
    variable_card=2, values=[
        [0.95, 0.1],
        [0.05, 0.9]
    ],
    evidence=['COVID'],
    evidence_card=[2]
)

model.add_cpds(cpd_fever, cpd_cough, cpd_travel, cpd_contact, cpd_covid,
               cpd_test_result)

assert model.check_model(), "Model is not consistent!"

infer = VariableElimination(model)

query = infer.query(
    variables=['COVID'],
    evidence={
        'Fever': 1,
        'Cough': 1,
        'TravelHistory': 1,
        'ContactWithInfected': 1
    }
)

print("◻ Probability of COVID given symptoms:")
print(query)
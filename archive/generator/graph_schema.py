import json

# Open and load the graph schema json file
with open('schema.json', 'r',encoding='utf-8-sig') as file:
    schema = json.load(file)
    
# Extract nodes and edges from the schema
nodes = [node['labels'][0] for node in schema[0]['nodes']]
edges = [relationship['type'] for relationship in schema[0]['relationships']]
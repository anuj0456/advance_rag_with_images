from services.llm_processor import LLMProcessor
from services.process_data import DataProcessor
from flask import Blueprint, request, jsonify, make_response


bp = Blueprint('app', __name__)

pllm = LLMProcessor()
dp = DataProcessor()


@bp.route('/qa', methods=['POST'])
def get_answer():
    query = request.get_json()['query']
    suffix_for_images = " Include any pie charts, graphs, or tables."
    docs = dp.get_multivector_store(query, suffix_for_images)

    data = {'data': docs, 'code': 'SUCCESS'}
    return make_response(jsonify(data), 200)

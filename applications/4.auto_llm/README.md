# 🚀🤖 AutoLLM: RAG-based LLM Apps and APIs ⏱️⚡

![image](https://github.com/user-attachments/assets/2a4186a1-ac90-49c3-a9e5-e669fad49cf0)


AutoLLM is a powerful toolkit designed to simplify the deployment, management, and integration of **Language Models (LLMs)** with multiple data sources, vector databases, and APIs. It enables you to build **RAG-based (Retrieval-Augmented Generation)** applications in **seconds**. Whether you're working with 100+ language models or need automatic cost calculations, AutoLLM provides the tools you need with minimal setup.

## Key Features ✨
- **100+ LLMs**: Seamlessly integrate and manage over 100 language models.
- **Unified API**: Use a single interface to handle all your LLMs.
- **20+ Vector Databases**: Support for vector databases such as LanceDB, and easy setup.
- **Cost Calculation**: Track token usage and cost calculations for each LLM query.
- **Fast API Deployment**: Create and deploy **FastAPI** apps with a single line of code.
- **One-Line Query Engine**: Instantly generate a query engine for any document or data source.

# 🚀 Quickstart


## Automated Cost Calculation 💰

AutoLLM can automatically calculate token usage and the associated cost for each query. Here’s how you can enable cost tracking:

from autollm import AutoServiceContext

# Enable cost calculation
service_context = AutoServiceContext(enable_cost_calculation=True)

- Example verbose output after a query
- Embedding Token Usage: 7
- LLM Prompt Token Usage: 1482
- LLM Completion Token Usage: 47
- LLM Total Token Cost: $0.002317


## FAQs ❓
### Can I use AutoLLM for commercial projects?
Yes, AutoLLM is licensed under the GNU Affero General Public License (AGPL 3.0), which allows commercial use under certain conditions. For more details, please contact us.

### Is AutoLLM easy to integrate into existing applications?
Yes! AutoLLM is designed for easy integration with your existing workflows. You can quickly migrate from other frameworks like Llama-Index and set up RAG-based applications with minimal effort.

### Installation

To install AutoLLM, simply run:

```bash
pip install autollm

# Use Slim Python base image
FROM python:3.11-slim

#  Setting  Working Directory Inside the contatiner
WORKDIR /app

# Copy dependencies files
COPY pyproject.toml uv.lock* ./

# install uv
RUN pip install uv
RUN uv sync --frozen --no-dev

# Copy Project Files
COPY . .

# Expose  Fast ApI to default port
EXPOSE 8000

# Command to Run API with unicorn
CMD [ "uv","run","unicorn","src.api.main:app","--host","0.0.0.0","--port","8000" ]
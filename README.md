# free-gpt35-api

```sh
# init project
rye sync

# run
rye run python src/free_gpt35_api/main.py
```

request 

```sh
curl -X POST -H "Content-Type: application/json" -d '{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "temperature": 0.7,
  "stream": true
}' http://localhost:8000/v1/chat/completions
```
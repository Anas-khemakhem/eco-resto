from app import create_app

app = create_app()

if __name__ == '__main__':
    # Professional pattern: Run via this entrypoint
    app.run(debug=True, port=5000)
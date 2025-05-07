CREATE TABLE users (
    user_id INT PRIMARY KEY,
    full_name VARCHAR(255),
    phone VARCHAR(50),
    email VARCHAR(255),
    created_at DATETIME,
    last_active_at DATETIME,
    is_vip BIT,
    total_balance DECIMAL(18,2)
);

CREATE TABLE cards (
    card_id INT PRIMARY KEY,
    user_id INT,
    card_number VARCHAR(20),
    balance DECIMAL(18,2),
    created_at DATETIME,
    card_type VARCHAR(50),
    limit_amount DECIMAL(18,2),
    exceeds_limit BIT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE transactions (
    transaction_id INT PRIMARY KEY,
    card_id INT,
    amount DECIMAL(18,2),
    status VARCHAR(50),
    transaction_date DATETIME,
    transaction_type VARCHAR(50),
    flag_large_amount BIT,
    FOREIGN KEY (card_id) REFERENCES cards(card_id)
);

CREATE TABLE logs (
    log_id INT PRIMARY KEY,
    transaction_id INT,
    message VARCHAR(MAX),
    event_timestamp DATETIME,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
);

CREATE TABLE reports (
    report_id INT PRIMARY KEY,
    report_type VARCHAR(50),
    generated_date DATETIME,
    total_transactions INT,
    flagged_transactions INT,
    total_amount DECIMAL(18,2)
);

CREATE TABLE scheduled_payments (
    schedule_id INT PRIMARY KEY,
    card_id INT,
    payment_amount DECIMAL(18,2),
    schedule_date DATETIME,
    status VARCHAR(50),
    FOREIGN KEY (card_id) REFERENCES cards(card_id)
);

CREATE TABLE retrieveinfo (
    retrieve_id INT IDENTITY(1,1) PRIMARY KEY,
    source_file VARCHAR(255),
    retrieved_at DATETIME,
    total_rows INT,
    processed_rows INT,
    errors INT,
    notes TEXT
);

select * from cards


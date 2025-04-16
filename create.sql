CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(256),
    base_price NUMERIC(7, 0) NOT NULL,
    discount NUMERIC(5, 2),
    rating NUMERIC(2, 1) DEFAULT 0.0,
    category VARCHAR(32),
    subcategory VARCHAR(32),
    brand VARCHAR(32),
    stock INT
);

CREATE TABLE spec_table (
    attr_name VARCHAR(50) NOT NULL,
    attr_value VARCHAR(200),
    product_id INT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
    PRIMARY KEY (attr_name, product_id)
);

CREATE TABLE images (
    product_id INT,
    img_url VARCHAR(256) NOT NULL,
    PRIMARY KEY (product_id, img_url),
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
);

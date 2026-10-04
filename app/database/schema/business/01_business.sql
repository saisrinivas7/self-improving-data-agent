-- ============================================================
-- Lumen & Co. business tables  (schema: public)
--
-- These are the ONLY tables the agent's generated SQL can see. Feedback
-- memory and traces live in the `memory` schema, which agent_ro cannot
-- reach - see 02_memory.sql for why that separation matters.
--
-- NO `COMMENT ON` STATEMENTS - THIS IS DELIBERATE
-- An earlier version of this file documented each column with COMMENT ON.
-- That was a mistake. Postgres stores those strings in pg_description, and
-- pg_description is readable by PUBLIC - so agent_ro could query them, and a
-- schema tool built to spec section 10 would paste them into every prompt.
-- The comments said things like "Net revenue = total_amount - refunds" and
-- "Not the order month", which is precisely the knowledge the Learning Lab
-- exists to teach the agent through analyst feedback. Leaving them in would
-- have handed the baseline the answers and quietly destroyed the experiment.
--
-- The `--` comments in this FILE are not stored in the database, so they
-- stay. A "documented schema" variant would make a reasonable later ablation:
-- does an agent with a well-documented schema still need feedback?
--
-- A note on the deliberate awkwardness in this schema: several columns are
-- shaped the way real systems are shaped rather than the way a tidy
-- benchmark would be. orders.total_amount is GROSS of refunds;
-- order_items.unit_price is what was actually paid while products.price is
-- list price; refunds.refund_date lags the order. Each of those is a trap
-- a naive text-to-SQL agent falls into, and each one is listed in
-- data/effects_manifest.yml with the mistake it produces. They exist so the
-- baseline agent makes genuine, correctable errors.
-- ============================================================

DROP TABLE IF EXISTS support_tickets CASCADE;
DROP TABLE IF EXISTS promotions      CASCADE;
DROP TABLE IF EXISTS refunds         CASCADE;
DROP TABLE IF EXISTS order_items     CASCADE;
DROP TABLE IF EXISTS orders          CASCADE;
DROP TABLE IF EXISTS products        CASCADE;
DROP TABLE IF EXISTS customers       CASCADE;

-- ------------------------------------------------------------ customers
CREATE TABLE customers (
    customer_id      integer PRIMARY KEY,
    name             text    NOT NULL,
    -- NULL for ~3% of customers. A naive GROUP BY country silently drops
    -- them from totals, which is one of the planted traps.
    country          text,
    signup_date      date    NOT NULL,
    customer_segment text    NOT NULL
                     CHECK (customer_segment IN ('Budget','Standard','Premium','VIP'))
);

-- ------------------------------------------------------------ products
CREATE TABLE products (
    product_id   integer PRIMARY KEY,
    product_name text    NOT NULL,
    category     text    NOT NULL,
    -- List price. NOT necessarily what any given order paid: see
    -- order_items.unit_price.
    price        numeric(10,2) NOT NULL CHECK (price > 0),
    -- Unit cost. Required for any margin question; easy to forget.
    cost         numeric(10,2) NOT NULL CHECK (cost > 0)
);


CREATE INDEX idx_products_category ON products (category);

-- ------------------------------------------------------------ orders
CREATE TABLE orders (
    order_id     integer PRIMARY KEY,
    customer_id  integer NOT NULL REFERENCES customers (customer_id),
    order_date   date    NOT NULL,
    -- completed | pending | cancelled | refunded
    -- cancelled + pending are ~12% of rows. Summing total_amount without
    -- filtering on status overstates revenue by roughly that much.
    status       text    NOT NULL
                 CHECK (status IN ('completed','pending','cancelled','refunded')),
    -- GROSS order value. Refunds are NOT subtracted here; they live in the
    -- refunds table. Net revenue requires joining to it.
    total_amount numeric(12,2) NOT NULL CHECK (total_amount >= 0)
);

CREATE INDEX idx_orders_date        ON orders (order_date);
CREATE INDEX idx_orders_customer    ON orders (customer_id);
CREATE INDEX idx_orders_status      ON orders (status);
CREATE INDEX idx_orders_date_status ON orders (order_date, status);

-- ------------------------------------------------------------ order_items
CREATE TABLE order_items (
    order_id   integer NOT NULL REFERENCES orders (order_id),
    product_id integer NOT NULL REFERENCES products (product_id),
    quantity   integer NOT NULL CHECK (quantity > 0),
    -- The price ACTUALLY PAID per unit, after any promotion. Differs from
    -- products.price during promo periods. Using products.price for revenue
    -- overstates promotional months.
    unit_price numeric(10,2) NOT NULL CHECK (unit_price > 0),
    PRIMARY KEY (order_id, product_id)
);

CREATE INDEX idx_order_items_product ON order_items (product_id);

-- ------------------------------------------------------------ refunds
CREATE TABLE refunds (
    refund_id     integer PRIMARY KEY,
    order_id      integer NOT NULL REFERENCES orders (order_id),
    -- Trails orders.order_date by 5-30 days, so a March order is often
    -- refunded in April. Grouping refunds by refund_date attributes them to
    -- the wrong month and smears the March effect.
    refund_date   date    NOT NULL,
    refund_amount numeric(12,2) NOT NULL CHECK (refund_amount > 0),
    -- late_delivery | damaged | wrong_item | not_as_described | changed_mind
    reason        text    NOT NULL
);

CREATE INDEX idx_refunds_order  ON refunds (order_id);
CREATE INDEX idx_refunds_date   ON refunds (refund_date);
CREATE INDEX idx_refunds_reason ON refunds (reason);

-- ------------------------------------------------------------ support_tickets
CREATE TABLE support_tickets (
    ticket_id   integer PRIMARY KEY,
    customer_id integer NOT NULL REFERENCES customers (customer_id),
    -- NULL when the ticket is not about a specific order.
    order_id    integer REFERENCES orders (order_id),
    created_at  timestamp NOT NULL,
    -- shipping_delay | damaged_item | billing | product_question | return_request | other
    category    text      NOT NULL,
    -- resolved | escalated | refunded | closed_no_action | open
    resolution  text      NOT NULL
);

CREATE INDEX idx_tickets_customer ON support_tickets (customer_id);
CREATE INDEX idx_tickets_order    ON support_tickets (order_id);
CREATE INDEX idx_tickets_created  ON support_tickets (created_at);
CREATE INDEX idx_tickets_category ON support_tickets (category);

-- ------------------------------------------------------------ promotions
CREATE TABLE promotions (
    promotion_id     integer PRIMARY KEY,
    product_id       integer NOT NULL REFERENCES products (product_id),
    start_date       date    NOT NULL,
    end_date         date    NOT NULL,
    discount_percent numeric(5,2) NOT NULL CHECK (discount_percent > 0 AND discount_percent < 100),
    CHECK (end_date >= start_date)
);

CREATE INDEX idx_promotions_product ON promotions (product_id);
CREATE INDEX idx_promotions_dates   ON promotions (start_date, end_date);

-- ------------------------------------------------------------ grants
-- agent_ro already has SELECT via ALTER DEFAULT PRIVILEGES in the initdb
-- script, but we grant explicitly so this file is self-contained and so
-- re-running it after a role change still works.
GRANT SELECT ON ALL TABLES IN SCHEMA public TO agent_ro;

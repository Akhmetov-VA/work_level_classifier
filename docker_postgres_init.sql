create table queries
(
    gender                  text,
    dob                     date,
    country                 text,
    district                text,
    region                  text,
    education               text,
    workplace               text    not null,
    jobname                 text    not null,
    inn                     text,
    okwed_type              text,
    okwed_1                 text,
    okwed_2                 text,
    okwed_3                 text,
    capital_type            text,
    capital_value           BIGINT,
    okogu_type              text,
    employee_cnt            BIGINT,
    smp_cat                 text,
    taxes_2022              BIGINT,
    scope_work              text,
    сareer_stage            text,
    row_id                  serial    primary key
);

alter table queries
    owner to postgres;

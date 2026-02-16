--
-- PostgreSQL database dump
--

-- Dumped from database version 14.17 (Homebrew)
-- Dumped by pg_dump version 14.17 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: clinical_data; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.clinical_data (
    clinical_id integer NOT NULL,
    patient_id integer NOT NULL,
    "timestamp" timestamp without time zone,
    timepoint integer,
    creatinine double precision,
    hemoglobin double precision,
    ldh integer,
    lymphocytes double precision,
    neutrophils double precision,
    platelet_count double precision,
    wbc_count double precision,
    hs_crp double precision,
    d_dimer double precision,
    news_score integer,
    news_score_label integer
);


ALTER TABLE public.clinical_data OWNER TO experi;

--
-- Name: clinical_data_clinical_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.clinical_data_clinical_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.clinical_data_clinical_id_seq OWNER TO experi;

--
-- Name: clinical_data_clinical_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.clinical_data_clinical_id_seq OWNED BY public.clinical_data.clinical_id;


--
-- Name: patient; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.patient (
    patient_id integer NOT NULL,
    patient_name character varying(100),
    severity integer,
    doctor_name character varying(100),
    hospital_name character varying(200),
    predicted_news integer DEFAULT 0
);


ALTER TABLE public.patient OWNER TO experi;

--
-- Name: patient_patient_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.patient_patient_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.patient_patient_id_seq OWNER TO experi;

--
-- Name: patient_patient_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.patient_patient_id_seq OWNED BY public.patient.patient_id;


--
-- Name: report; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.report (
    report_id integer NOT NULL,
    patient_id integer NOT NULL,
    from_hospital character varying(200),
    to_hospital character varying(200),
    context character varying(500),
    createdat timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    reservedat timestamp without time zone
);


ALTER TABLE public.report OWNER TO experi;

--
-- Name: report_report_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.report_report_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.report_report_id_seq OWNER TO experi;

--
-- Name: report_report_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.report_report_id_seq OWNED BY public.report.report_id;


--
-- Name: clinical_data clinical_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.clinical_data ALTER COLUMN clinical_id SET DEFAULT nextval('public.clinical_data_clinical_id_seq'::regclass);


--
-- Name: patient patient_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.patient ALTER COLUMN patient_id SET DEFAULT nextval('public.patient_patient_id_seq'::regclass);


--
-- Name: report report_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.report ALTER COLUMN report_id SET DEFAULT nextval('public.report_report_id_seq'::regclass);


--
-- Data for Name: clinical_data; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.clinical_data (clinical_id, patient_id, "timestamp", timepoint, creatinine, hemoglobin, ldh, lymphocytes, neutrophils, platelet_count, wbc_count, hs_crp, d_dimer, news_score, news_score_label) FROM stdin;
\.


--
-- Data for Name: patient; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.patient (patient_id, patient_name, severity, doctor_name, hospital_name) FROM stdin;
1	김민우	3	박철수	SKALA대학병원
2	이서현	3	김지훈	SKALA대학병원
3	박지훈	2	이영희	SKALA대학병원
4	최유진	2	정우성	SKALA대학병원
5	정하늘	6	한지민	SKALA대학병원
6	한도윤	4	최민호	SKALA대학병원
7	윤지호	6	오하늘	SKALA대학병원
8	서지민	7	김도현	SKALA대학병원
9	장예린	5	송혜교	SKALA대학병원
10	오승현	8	김범수	SKALA대학병원
\.


--
-- Data for Name: report; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.report (report_id, patient_id, from_hospital, to_hospital, context, createdat, reservedat) FROM stdin;
\.


--
-- Name: clinical_data_clinical_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.clinical_data_clinical_id_seq', 1, false);


--
-- Name: patient_patient_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.patient_patient_id_seq', 1, false);


--
-- Name: report_report_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.report_report_id_seq', 1, false);


--
-- Name: clinical_data clinical_data_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.clinical_data
    ADD CONSTRAINT clinical_data_pkey PRIMARY KEY (clinical_id);


--
-- Name: patient patient_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.patient
    ADD CONSTRAINT patient_pkey PRIMARY KEY (patient_id);


--
-- Name: report report_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.report
    ADD CONSTRAINT report_pkey PRIMARY KEY (report_id);


--
-- Name: clinical_data fk_clinical_patient; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.clinical_data
    ADD CONSTRAINT fk_clinical_patient FOREIGN KEY (patient_id) REFERENCES public.patient(patient_id) ON DELETE CASCADE;


--
-- Name: report fk_report_patient; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.report
    ADD CONSTRAINT fk_report_patient FOREIGN KEY (patient_id) REFERENCES public.patient(patient_id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--


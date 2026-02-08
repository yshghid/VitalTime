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
    hospital_name character varying(200)
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
70	7	2025-01-04 21:46:21	10	0.69	14.37	215	2.67	4.44	299	6.13	4.2	0.69	6	8
1	1	2025-01-01 23:25:05	1	0.9	14.1	180	2.1	4.2	220	7	1.8	0.4	2	2
2	1	2025-01-02 07:25:05	2	1.1	14	210	2	4.5	210	7.3	2	0.6	3	4
3	1	2025-01-02 15:25:05	3	1.3	13.8	240	1.8	5	200	8.1	3.5	0.8	4	4
4	1	2025-01-02 23:25:05	4	1	14.3	200	2.3	4	230	6.5	1.5	0.5	2	4
5	1	2025-01-03 07:25:05	5	1.5	13.7	260	1.7	5.3	190	8.7	4.2	1	5	7
6	1	2025-01-03 15:25:05	6	1.2	13.9	230	1.9	4.8	205	7.9	3	0.7	3	3
7	1	2025-01-03 23:25:05	7	1.6	13.6	280	1.6	5.5	185	9	4.5	1.2	5	4
8	1	2025-01-04 07:25:05	8	1	14.2	190	2.2	4.1	225	6.8	1.7	0.4	2	4
9	1	2025-01-04 15:25:05	9	1.4	13.8	250	1.8	5.1	195	8.4	3.8	0.9	4	2
10	1	2025-01-04 23:25:05	10	1.2	14	220	2	4.6	210	7.5	2.2	0.6	3	3
41	5	2025-01-01 02:27:18	1	1	14.4	223	1.24	5.88	246	7.9	4.38	0.21	3	1
42	5	2025-01-01 10:27:18	2	1.54	12.7	207	1.99	5.1	226	5	3.3	0.72	5	3
43	5	2025-01-01 18:27:18	3	1.43	12.3	236	1.07	5.76	321	4	1.99	0.77	5	6
44	5	2025-01-02 02:27:18	4	1.38	15.8	260	2.82	5.58	241	8.1	0.79	0.47	5	5
45	5	2025-01-02 10:27:18	5	1.01	15.9	287	1.52	4.39	228	7.5	1.9	0.52	5	5
46	5	2025-01-02 18:27:18	6	1.05	15.2	339	2.33	5.69	272	7.6	1.96	0.58	5	5
71	8	2025-01-01 08:35:14	1	0.9	14.1	180	2.2	4.5	230	7.1	3.5	0.6	5	7
72	8	2025-01-01 16:35:14	2	1.2	13.8	250	1.9	5.8	210	8	6.2	0.9	6	4
73	8	2025-01-02 00:35:14	3	1.6	13.2	310	1.7	6.2	200	8.5	9.5	1.4	8	10
74	8	2025-01-02 08:35:14	4	2	12.9	360	1.5	7	190	9.2	12	2	9	9
75	8	2025-01-02 16:35:14	5	1.7	12.5	330	1.6	6.5	185	8.8	10.5	1.6	8	10
76	8	2025-01-03 00:35:14	6	2.2	12	410	1.4	7.5	170	10	14.2	2.5	10	9
77	8	2025-01-03 08:35:14	7	1.8	12.8	290	1.8	6	195	9	8	1.2	7	7
78	8	2025-01-03 16:35:14	8	1.4	13	240	2	5.2	210	8.2	5	0.8	6	8
61	7	2025-01-01 21:46:21	1	0.78	13.16	188	1.7	2.4	187	8.36	5.97	0.46	4	5
62	7	2025-01-02 05:46:21	2	1.53	14.45	238	2.03	2.88	312	4.87	5.28	0.89	7	8
63	7	2025-01-02 13:46:21	3	1.38	12.56	244	2.07	2.2	174	6.29	7.15	0.47	5	5
64	7	2025-01-02 21:46:21	4	0.95	13.17	235	1.5	3.46	348	5.08	4.04	0.24	5	5
65	7	2025-01-03 05:46:21	5	0.39	13.47	198	2.75	3.75	307	8.82	1.4	0.37	4	4
66	7	2025-01-03 13:46:21	6	0.81	13.82	187	2.44	3.22	198	7.62	5.85	0.8	6	8
67	7	2025-01-03 21:46:21	7	1.32	15.14	150	2.7	5.73	161	6.15	6.21	0.75	5	4
68	7	2025-01-04 05:46:21	8	1.2	12.8	149	2.63	3.61	315	4.82	4.71	0.13	3	4
69	7	2025-01-04 13:46:21	9	0.31	14.06	199	2.16	3.26	294	6.05	6.28	0.06	3	1
47	5	2025-01-03 02:27:18	7	1	13.2	297	1.62	2.35	204	7.9	3.78	0.75	5	3
48	5	2025-01-03 10:27:18	8	1.93	12.4	350	2.04	2.78	316	4.4	3.37	1.05	7	9
49	5	2025-01-03 18:27:18	9	1.71	14.7	379	2.09	2.18	193	5.8	4.49	1.04	6	8
50	5	2025-01-04 02:27:18	10	1.9	13.8	330	1.37	3.3	348	4.6	2.62	0.95	6	7
11	2	2025-01-01 03:12:39	1	0.9	14.5	180	2.2	4.5	230	7	1.8	0.4	1	2
12	2	2025-01-01 11:12:39	2	1.1	14.2	220	2	4.8	210	7.5	3.5	0.6	2	1
13	2	2025-01-01 19:12:39	3	1.4	13.8	250	1.9	5.2	200	8	5.2	0.8	3	1
14	2	2025-01-02 03:12:39	4	1.7	13.5	300	1.6	5.8	190	8.5	7.8	1.2	4	3
15	2	2025-01-02 11:12:39	5	1.5	13.9	280	2.1	5	210	7.9	6.5	1	3	4
16	2	2025-01-02 19:12:39	6	1.8	13.4	350	1.5	6.2	180	9	10.2	1.5	5	5
17	2	2025-01-03 03:12:39	7	1.6	13.7	310	1.8	5.6	200	8.2	8.7	1.1	4	2
18	2	2025-01-03 11:12:39	8	1.2	14	240	2.3	4.7	220	7.2	4.8	0.7	2	3
19	2	2025-01-03 19:12:39	9	1	14.3	200	2.5	4.4	240	6.8	2.5	0.5	1	1
20	2	2025-01-04 03:12:39	10	1.3	13.9	270	2	5.1	210	7.6	5.9	0.9	3	2
31	4	2025-01-01 17:37:04	1	0.9	14	180	2	4.5	250	7	2.5	0.3	1	1
32	4	2025-01-02 01:37:04	2	1.2	13.5	220	1.8	5	240	7.5	3.2	0.6	2	4
33	4	2025-01-02 09:37:04	3	1.8	13	300	1.6	6	220	8	5.5	0.9	3	1
34	4	2025-01-02 17:37:04	4	2.1	12.5	380	1.5	6.5	210	9	7	1.2	4	4
35	4	2025-01-03 01:37:04	5	1.5	12.8	250	1.9	5.5	230	7.5	4.5	0.7	2	4
36	4	2025-01-03 09:37:04	6	2.4	12.2	420	1.3	7	200	10	9.8	1.5	5	7
37	4	2025-01-03 17:37:04	7	1.7	12.5	280	1.7	5.8	215	8.5	6	0.8	3	2
38	4	2025-01-04 01:37:04	8	0.8	13.8	190	2.2	4.8	260	7.2	2	0.4	1	1
39	4	2025-01-04 09:37:04	9	2	12.3	360	1.4	6.8	205	9.5	8.2	1.1	4	2
40	4	2025-01-04 17:37:04	10	1.3	13	230	1.8	5.3	240	8	3.8	0.6	2	1
51	6	2025-01-01 03:03:08	1	1.37	11.1	425	2.52	2.85	392	7.11	15.56	2.62	7	6
52	6	2025-01-01 11:03:08	2	2.41	15.8	213	1.43	5.47	344	6.17	4.37	1.95	5	7
53	6	2025-01-01 19:03:08	3	2.02	15.2	281	1.16	2.24	385	10.63	0.61	1.13	5	5
79	8	2025-01-04 00:35:14	9	1	13.5	190	2.3	4.8	220	7.5	3	0.5	5	6
80	8	2025-01-04 08:35:14	10	1.5	13.3	280	1.9	5.6	205	8.4	7	1	7	9
81	9	2025-01-01 16:06:16	1	1.03	12.93	537	1.81	4.48	228.38	11.1	9.87	0.87	7	6
82	9	2025-01-02 00:06:16	2	1.8	15.9	599	3.4	4.27	198.93	11.51	24.9	3.84	10	10
83	9	2025-01-02 08:06:16	3	1.46	15.72	279	2.3	6.93	229.13	11.1	39.08	2.52	8	8
84	9	2025-01-02 16:06:16	4	1.1	15.89	326	3.02	3.67	294.21	6.95	25.61	1.22	6	4
85	9	2025-01-03 00:06:16	5	1.91	12.3	339	2.32	3.13	236.49	7.24	4.27	4.11	9	8
86	9	2025-01-03 08:06:16	6	1.04	13.62	292	0.99	7.77	216.21	4.49	8.48	2.19	6	4
87	9	2025-01-03 16:06:16	7	0.95	13.07	524	2.66	4.65	293.46	9.63	7.04	3.78	7	9
88	9	2025-01-04 00:06:16	8	1.06	11.19	389	1.8	2.67	220.6	5.05	12.17	0.78	6	4
89	9	2025-01-04 08:06:16	9	1.03	11.67	301	2.12	7.27	156.05	7.02	4.52	3.16	7	9
90	9	2025-01-04 16:06:16	10	0.72	13.64	196	2.08	7.27	192.44	8.41	25.93	2.06	5	7
91	10	2025-01-01 21:34:06	1	1.55	13.6	209	1.88	8.27	300	10.58	11.3	1.86	7	8
92	10	2025-01-02 05:34:06	2	1.55	11.5	413	2	7.06	336	5.23	7.4	3.92	10	10
93	10	2025-01-02 13:34:06	3	1.27	13.1	485	2.86	3.28	327	4.06	4	4.08	10	10
94	10	2025-01-02 21:34:06	4	1.33	13.8	156	1.43	2.29	239	5.71	7.1	1.12	7	6
95	10	2025-01-03 05:34:06	5	1.22	13.1	364	2.66	2.54	270	6.57	15.7	3.95	10	10
96	10	2025-01-03 13:34:06	6	1.44	14.5	318	2.61	2.45	244	6.65	0.8	3.37	10	8
97	10	2025-01-03 21:34:06	7	0.86	15.3	318	2.78	4.09	299	4.82	5.8	3.92	10	9
98	10	2025-01-04 05:34:06	8	0.97	14.2	480	2.05	2.78	400	9.84	16	4.99	10	9
99	10	2025-01-04 13:34:06	9	1.09	15.9	195	1.17	4.88	285	6.11	4.9	1.68	6	6
100	10	2025-01-04 21:34:06	10	1.43	14.6	172	2.2	8.93	309	4.67	5.3	2.61	8	8
54	6	2025-01-02 03:03:08	4	1.78	12.1	315	3.37	8.37	374	6.85	16.4	0.38	4	3
55	6	2025-01-02 11:03:08	5	0.98	11.9	355	3.41	3.81	299	6.25	14.28	1.07	3	4
56	6	2025-01-02 19:03:08	6	0.98	11.9	503	3.02	6.64	380	8.34	14.72	1.11	5	7
57	6	2025-01-03 03:03:08	7	0.8	12.5	240	1.76	4.18	172	5.13	15.54	2.24	3	3
58	6	2025-01-03 11:03:08	8	2.26	13.6	381	1.24	5.64	199	10.42	1.94	1.99	7	7
59	6	2025-01-03 19:03:08	9	1.78	13.2	417	2.71	5.83	161	4.6	7.49	2.68	7	5
60	6	2025-01-04 03:03:08	10	1.97	12.5	171	2.1	3.29	231	11.9	2.76	1.52	4	6
21	3	2025-01-01 14:27:28	1	0.9	14.2	180	2	4.5	220	7	2.1	0.4	1	1
22	3	2025-01-01 22:27:28	2	1.1	14	220	2.3	5	210	7.5	3	0.6	2	2
23	3	2025-01-02 06:27:28	3	1.4	13.8	260	1.8	6	200	8.5	5.2	0.8	3	3
24	3	2025-01-02 14:27:28	4	1.2	13.6	230	2.5	5.2	240	7.8	4.8	0.7	2	4
25	3	2025-01-02 22:27:28	5	1.6	13.5	280	1.6	6.5	190	9	7.5	1	4	6
26	3	2025-01-03 06:27:28	6	1.3	13.7	240	2.1	5.8	230	8.2	4	0.9	3	5
27	3	2025-01-03 14:27:28	7	1.8	13.4	310	1.4	6.8	180	9.5	8.2	1.5	5	4
28	3	2025-01-03 22:27:28	8	1.5	13.9	260	2	6	210	8	5	1.1	3	5
29	3	2025-01-04 06:27:28	9	1.7	13.6	300	1.7	7	200	9.2	7.8	1.3	4	5
30	3	2025-01-04 14:27:28	10	1.2	14.1	210	2.4	5.5	225	7.6	3.5	0.5	2	1
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

SELECT pg_catalog.setval('public.clinical_data_clinical_id_seq', 10, true);


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


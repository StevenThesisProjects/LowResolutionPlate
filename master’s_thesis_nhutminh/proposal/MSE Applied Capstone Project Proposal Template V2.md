# **MASTER OF SOFTWARE ENGINEERING (MSE)**

## **CAPSTONE PROJECT PROPOSAL TEMPLATE**

**Note for Students:** This template is designed specifically for an *applied software engineering capstone project*. The focus must be on solving a real-world business or technical problem through rigorous software engineering practices, system design, and practical implementation rather than purely theoretical research.

## ---

**1\. INTRODUCTION**

### **1.1 Background**

Describe the operational domain, business environment, or industry sector where this project is positioned. Identify the stakeholders, end-users, or organizational units affected by the current state of technology or processes within this domain.

### **1.2 Practical problem statement and User pain-point**

Clearly and specifically articulate the real-world problem or technical inefficiency your project aims to solve. Avoid generic or theoretical problem statements; instead, anchor the issue in concrete operational impacts (e.g., scalability bottlenecks, data synchronization delays, manual processing overheads, or security vulnerabilities).

### **1.3 Research Objectives / Questions**

Define the specific, measurable, achievable, relevant, and time-bound (SMART) engineering objectives that the project will deliver. Frame these goals around concrete software metrics or business capabilities rather than abstract research milestones.

### **1.4 Scope**

Establish the exact boundaries of the project. Clearly differentiate what will be built from what will be omitted to ensure feasibility within the capstone timeline.

| In-Scope Components | Out-of-Scope Components   |
| :---- | :---- |
| List the core modules, platforms, APIs, and user roles that will be fully implemented and tested. | List legacy integrations, third-party systems assumed to be black boxes, or secondary features deferred to future releases. |

## **2\. LITERATURE AND TECHNICAL REVIEW**

### **2.1 Overview of existing solutions**

Analyze existing commercial systems, open-source libraries, or industrial frameworks that address similar problems. Evaluate their strengths and operational trade-offs from a practical engineering perspective.

### **2.2 Technical gaps / Limitations of current approaches**

Identify why existing solutions are insufficient for your specific problem context. Focus on concrete architectural limitations, cost constraints, licensing barriers, performance bottlenecks, or unique business requirements that remain unfulfilled.

### **2.3 Rationale for the selected approach and stack**

Justify your choice of architecture, programming languages, databases, cloud providers, and development frameworks. The rationale must be backed by engineering trade-off analyses (e.g., comparing SQL vs. NoSQL, or Microservices vs. Monolithic architectures for your specific workload).

## **3\. PROPOSED SOLUTION AND ENGINEERING DESIGN**

### **3.1 High-level system architecture**

Provide a conceptual overview of the system architecture (e.g., Microservices, Event-Driven, Cloud-Native, Serverless, or Monolithic). Students should include high-level block diagrams detailing user access points, application layers, data persistence layers, and external network interactions.

### **3.2 Core component Design and Interfacing**

Detail the primary software subsystems, data models, or service components that compose the solution. Specify the communication protocols (e.g., RESTful APIs, gRPC, WebSockets, or Message Queues) that enable data exchange between these components.

### **3.3 Software Engineering Methodology and DevOps Pipeline**

Outline the development methodology (e.g., Agile Scrum, Kanban) to be practiced during execution. Detail the DevOps infrastructure, including Continuous Integration/Continuous Deployment (CI/CD) pipelines, automated linting, containerization strategies (e.g., Docker), and cloud orchestration tools.

## **4\. EXPECTED RESULTS AND PERFORMANCE EVALUATION**

### **4.1 Deliverables**

Enumerate the concrete artifacts that will be delivered upon project completion. Every applied project must output actionable codebases and operational environments.

> * Production-ready source code repository with comprehensive README instructions.  
> * Automated deployment configuration scripts (e.g., Terraform, Kubernetes manifests, Docker Compose files).  
> * Comprehensive API documentation (e.g., Swagger/OpenAPI specifications).  
> * User manuals and architectural documentation.

### **4.2 Verification and Validation Plan**

Detail the automated and manual testing strategies designed to guarantee system reliability, performance, and compliance with requirements.

Some examples:

| Testing Tier | Methodology and Tools | Success Criteria / Target Metrics   |
| :---- | :---- | :---- |
| **Unit Testing** | Automated framework runs (e.g., JUnit, PyTest). | Minimum 80% statement and branch coverage. |
| **Integration Testing** | End-to-end API path flows and database transitions. | Zero critical path connection dropouts. |
| **Performance Testing** | Load testing under high traffic simulations (e.g., JMeter, Locust). | P95 response time \< 200ms under 1000 concurrent req/sec. |
| **Security Testing** | Static Application Security Testing (SAST) and vulnerability scans. | Zero high or critical OWASP Top 10 vulnerabilities. |

### **4.3 Success Metrics and SLA Compliance**

Define how the final product will be evaluated against business expectations. Identify Key Performance Indicators (KPIs) such as system uptime, query optimization percentages, transactional throughput, or user acceptance score thresholds.

## **5\. PROJECT IMPLEMENTATION AND RESEARCH PLAN**

### **5.1 Work Breakdown Structure (WBS) and Project phases**

Deconstruct the engineering lifecycle into manageable phases. Common phases for applied software engineering projects include: Requirements Elicitation & Analysis, Architectural & Database Design, Core Implementation Sprints, Testing & Security Auditing, and System Deployment.

### **5.2 Project timeline and Milestones**

Map the project schedule, highlighting critical milestones (e.g., completion of database schema, MVP release, full system test cycle, and final production deployment). Include a placeholder for a Gantt chart detailing parallel development workstreams.

### **5.3 Feasibility assessment and Risk mitigation Strategy**

Evaluate operational, technical, resource, and scheduling risks that could jeopardize project success. Provide actionable mitigation strategies for each identified threat.

| Risk Category | Identified Threat Description | Actionable Mitigation Strategy   |
| :---- | :---- | :---- |
| **Technical Risk** | Complexity or steep learning curve of a newly adopted framework or cloud service. | Build an isolated Proof of Concept (PoC) during Week 1-2 to validate core integrations. |
| **Resource Risk** | Lack of budget or permissions for proprietary APIs or enterprise cloud infrastructure. | Leverage open-source alternatives (e.g., LocalStack for AWS local emulation, community tiers). |
| **Schedule Risk** | Scope creep causing delays in core deliverable completions. | Implement strict feature prioritization using the MoSCoW method (Must have, Should have, etc.). |

## **6\. REFERENCES**

List all industry papers, technical documentation, whitepapers, structural benchmarks, and relevant academic publications consulted during proposal formulation. 

**Format:** Use the **IEEE reference style** for technical consistency across software engineering artifacts.
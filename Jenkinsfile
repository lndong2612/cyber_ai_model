pipeline {
    agent any
    stages {
        stage('Init') {
            steps {
                echo 'Testing..'
                telegramSend(message: 'Building job cybersecurity ai bca...', chatId: -740504133)
            }
        }
        stage ('Deployments') {
            steps {
                // && rancher kubectl apply -f ./thinklabsdev/cybersecurityaiBCACI/cybersecuritybca_pvc.yaml \
                echo 'Deploying to Production environment...'
                echo 'Copy project over SSH...'
                sshPublisher(publishers: [
                    sshPublisherDesc(
                        configName: 'swarm1',
                        transfers:
                            [sshTransfer(
                                cleanRemote: false,
                                excludes: '',
                                execCommand: """docker build -t registry.thinklabs.com.vn:5000/cybersecurityai:bca ./thinklabsdev/cybersecurityaiBCACI/ \
                                    && docker image push registry.thinklabs.com.vn:5000/cybersecurityai:bca \
                                    && if [ ! -f ".rancher/cli2.json" ]; then rancher login https://k8smanager.thinklabs.com.vn --token token-7cwpc:9dccvb5qc6jkv2529878spz5xvd74sntftn9nzmnxlj96s9q645jgt --context local:p-ldw6l; else rancher context switch local:p-ldw6l; fi \
                                    && rancher namespaces ls | grep "cybersecuritybca" || rancher namespaces create cybersecuritybca \
                                    && rancher kubectl apply -f ./thinklabsdev/cybersecurityaiBCACI/cybersecurityaiBCAService.yaml \
                                    && rancher kubectl apply -f ./thinklabsdev/cybersecurityaiBCACI/cybersecurityaiBCAIngress.yaml \
                                    && rancher kubectl delete -f ./thinklabsdev/cybersecurityaiBCACI/cybersecurityaiBCA_deployment.yaml || true \
                                    && rancher kubectl apply -f ./thinklabsdev/cybersecurityaiBCACI/cybersecurityaiBCA_deployment.yaml \
                                    && rm -rf ./thinklabsdev/cybersecurityaiBCACIB \
                                    && mv ./thinklabsdev/cybersecurityaiBCACI/ ./thinklabsdev/cybersecurityaiBCACIB""",                                                              
                                execTimeout: 6000000,
                                flatten: false,
                                makeEmptyDirs: false,
                                noDefaultExcludes: false,
                                patternSeparator: '[, ]+',
                                remoteDirectory: './thinklabsdev/cybersecurityaiBCACI',
                                remoteDirectorySDF: false,
                                removePrefix: '',
                                sourceFiles: '*, app/, public/, ssl/'
                            )],
                        usePromotionTimestamp: false,
                        useWorkspaceInPromotion: false,
                        verbose: false
                    )
                ])
                telegramSend(message: 'Build Job cybersecurity ai bca -STATUS: $BUILD_STATUS!', chatId: -740504133)
            }
        }
    }
}
